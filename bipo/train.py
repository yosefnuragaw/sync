"""Train BiPO steering vectors for one behaviour, keeping the best-validation epoch.

Every wrapped layer carries a trainable vector; the trainer flips the injection sign from
batch to batch and flips the DPO objective with it, so one vector steers toward the
`matching` continuation at +1 and the `not_matching` one at -1. The held-out split is
scored in both directions each epoch and ranked by the mean.

Outputs, under `{vec_dir}`:
    pos_layer_{i}.pt / neg_layer_{i}.pt   best epoch, CAA layout
    bipo_train_report.json                per-epoch losses, epoch kept, per-layer norms

Usage (from the repo root, where ./data is read from):
    python bipo/train.py --config bipo/configs/train/llama3_1/power.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)

BIPO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BIPO_ROOT.parent
sys.path.insert(0, str(BIPO_ROOT))   # the vendored trl that holds BiPOTrainer
sys.path.append(str(REPO_ROOT))      # utils/, shared with billy/

from trl import BiPOTrainer, DPOConfig  # noqa: E402
from utils.common import get_data, print_trainable_parameters, set_seed  # noqa: E402
from wrapper import BiPOBlockWrapper  # noqa: E402

SEED = 42

# The two eval passes BiPOTrainer runs per epoch. It picks the injection sign off the
# dataset name ('add' -> +1, 'sub' -> -1), so these keys are not free-form.
EVAL_DATASETS = ("add", "sub")
VECTOR_DTYPE = torch.bfloat16  # what billy/extract_caa.py writes, so the stacks match


@dataclass
class ScriptArguments:
    id: str = field(default="bipo", metadata={"help": "Run id, printed and stored in the report"})
    model_name_or_path: str = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint the vectors are trained against"},
    )
    behavior: str = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: List[int] = field(
        default_factory=lambda: list(range(32)),
        metadata={"help": "Layers that carry a trainable steering vector"},
    )
    total_layer: int = field(default=32, metadata={"help": "LLM total number of layers"})
    vec_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory the best-epoch vectors and the training report are written to"},
    )

    # DPO objective
    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    learning_rate: float = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: int = field(default=20, metadata={"help": "the number of warmup steps"})
    weight_decay: float = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: str = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: int = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "gradient accumulation steps"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "use gradient checkpointing"})

    max_prompt_length: int = field(default=2048, metadata={"help": "maximum prompt length"})
    max_length: int = field(default=2048, metadata={"help": "maximum sequence length"})
    num_train_epochs: int = field(default=20, metadata={"help": "number of training epochs"})
    logging_steps: int = field(default=10, metadata={"help": "logging frequency"})

    val_fraction: float = field(
        default=0.1,
        metadata={
            "help": "Share of train.csv held out for validation, used only for behaviours that "
            "ship no data/{behavior}/test.csv (the persona-2 ones)"
        },
    )


def require_repo_root() -> None:
    """utils.common.get_data reads ./data/{behaviour}, so the cwd has to be the repo root."""
    if not Path("data").is_dir():
        raise SystemExit(
            f"run this from the repo root ({REPO_ROOT}); the data loader reads ./data/<behaviour>"
        )


def resolve_vec_dir(vec_dir: Optional[str]) -> str:
    """Where the learned vectors are written, as the config gives it."""
    if not vec_dir:
        raise ValueError("vec_dir must be set in the config file")
    return vec_dir


def load_datasets(tokenizer: PreTrainedTokenizerBase, behavior: str, val_fraction: float):
    """Return (train, validation, provenance) preference splits for one behaviour.

    data/{behaviour}/test.csv is the validation split wherever it exists. Where a
    behaviour ships only train.csv and test_infer.csv -- which has no chosen/rejected
    pair to score a DPO loss on -- a seeded slice of train.csv is held out instead.
    """
    train_dataset = get_data(tokenizer=tokenizer, behavior=behavior, train=True)

    if (REPO_ROOT / "data" / behavior / "test.csv").exists():
        eval_dataset = get_data(tokenizer=tokenizer, behavior=behavior, train=False)
        return train_dataset, eval_dataset, "data/%s/test.csv" % behavior

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"data/{behavior}/test.csv does not exist, so val_fraction must be in (0, 1) to carve "
            f"a validation split out of train.csv; got {val_fraction}"
        )
    split = train_dataset.train_test_split(test_size=val_fraction, seed=SEED, shuffle=True)
    return split["train"], split["test"], f"{val_fraction:.0%} holdout from data/{behavior}/train.csv"


def build_model(
    model_name: str,
    layers: List[int],
    total_layer: int,
) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the policy (steering vectors attached, base weights frozen) and its reference."""
    def load() -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.warnings_issued = {}
        model.config.use_cache = False
        return model

    model = load()
    available = len(model.model.layers)
    if not 0 < total_layer <= available:
        raise ValueError(f"total_layer must be in 1..{available} for {model_name}, got {total_layer}")
    out_of_range = [layer for layer in layers if not 0 <= layer < total_layer]
    if out_of_range:
        raise ValueError(f"layer {out_of_range} outside 0..{total_layer - 1}")

    for layer in layers:
        model.model.layers[layer] = BiPOBlockWrapper(
            model.model.layers[layer],
            hidden_dim=model.config.hidden_size,
        )

    # Only the steering vectors move; policy and reference are both frozen.
    ref_model = load()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in layers:
        model.model.layers[layer].vec.requires_grad_(True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, ref_model, tokenizer


class BestVectorSaver(TrainerCallback):
    """Write the steering vectors from the epoch with the lowest validation loss.

    The held-out split is scored twice per epoch, with the vector added and subtracted,
    and an epoch is ranked by the mean of the two, since generation reads `-v` off disk
    as the negative steering vector. Nothing is written until both directions have
    reported; the report is rewritten every epoch, so a run that dies part-way keeps its
    history.
    """

    def __init__(self, model: PreTrainedModel, layers: List[int], vec_dir: str, metadata: dict):
        self.model = model
        self.layers = layers
        self.vec_dir = Path(vec_dir)
        self.report_path = self.vec_dir / "bipo_train_report.json"
        self.metadata = metadata

        self.history: list[dict] = []
        self.best_loss = math.inf
        self.best_epoch: Optional[int] = None
        self.vector_norms: dict[str, float] = {}
        self._pending: dict[str, float] = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        for name in EVAL_DATASETS:
            key = f"eval_{name}_loss"
            if key in metrics:
                self._pending[name] = float(metrics[key])
        if len(self._pending) < len(EVAL_DATASETS):
            return  # the other direction has not been evaluated yet

        losses = {name: self._pending.pop(name) for name in EVAL_DATASETS}
        epoch = int(round(state.epoch or 0))
        val_loss = sum(losses.values()) / len(losses)
        improved = val_loss < self.best_loss

        if improved:
            self.best_loss, self.best_epoch = val_loss, epoch
            self.save_vectors()

        self.history.append({"epoch": epoch, "val_loss": val_loss, **losses, "saved": improved})
        directions = "  ".join(f"{name}={loss:.6f}" for name, loss in losses.items())
        marker = "  <- saved" if improved else f"  (best {self.best_loss:.6f} @ epoch {self.best_epoch})"
        print(f"[bipo] epoch {epoch:>3}  val_loss={val_loss:.6f}  {directions}{marker}")
        self.write_report()

    def save_vectors(self) -> None:
        self.vec_dir.mkdir(parents=True, exist_ok=True)
        norms = {}
        for layer in self.layers:
            vec = self.model.model.layers[layer].vec.detach().to(VECTOR_DTYPE).cpu()
            norms[str(layer)] = float(vec.float().norm())
            torch.save(vec, self.vec_dir / f"pos_layer_{layer}.pt")
            torch.save(-vec, self.vec_dir / f"neg_layer_{layer}.pt")
        self.vector_norms = norms  # of the epoch on disk, not of the latest one

    def write_report(self) -> None:
        self.vec_dir.mkdir(parents=True, exist_ok=True)
        report = dict(self.metadata)
        report.update(
            {
                "best_epoch": self.best_epoch,
                "best_val_loss": None if self.best_epoch is None else self.best_loss,
                "vector_norms": self.vector_norms,
                "history": self.history,
            }
        )
        with open(self.report_path, "w") as handle:
            json.dump(report, handle, indent=2)


def parse_script_arguments() -> ScriptArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to a .yaml or .json config")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warning] ignoring unrecognised arguments: {unknown}")

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        return hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    if args.config.endswith(".json"):
        return hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    raise ValueError("Config file must be .yaml or .json")


def main() -> None:
    require_repo_root()
    set_seed(seed=SEED)

    args = parse_script_arguments()
    vec_dir = resolve_vec_dir(args.vec_dir)

    print(f"[{args.id}] behavior={args.behavior} model={args.model_name_or_path}")
    print(f"[{args.id}] layers={args.layer}")
    print(f"[{args.id}] vectors -> {vec_dir}")

    model, ref_model, tokenizer = build_model(
        model_name=args.model_name_or_path,
        layers=args.layer,
        total_layer=args.total_layer,
    )
    print_trainable_parameters(model)

    train_dataset, eval_dataset, val_source = load_datasets(
        tokenizer=tokenizer, behavior=args.behavior, val_fraction=args.val_fraction
    )
    print(f"[{args.id}] train pairs: {len(train_dataset)}, validation pairs: {len(eval_dataset)} ({val_source})")

    training_args = DPOConfig(
        # save_strategy "no": the only artifacts are the vectors and the report the
        # callback writes into vec_dir.
        output_dir=os.path.join(vec_dir, "_trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        optim=args.optimizer_type,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="epoch",
        remove_unused_columns=False,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        beta=args.beta,
        bf16=torch.cuda.is_available(),
        report_to="none",  # tracking is local: stdout + bipo_train_report.json
        seed=SEED,
    )

    saver = BestVectorSaver(
        model=model,
        layers=args.layer,
        vec_dir=vec_dir,
        metadata={
            "id": args.id,
            "behavior": args.behavior,
            "model_name_or_path": args.model_name_or_path,
            "layers": args.layer,
            "beta": args.beta,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "validation_split": val_source,
            "train_pairs": len(train_dataset),
            "validation_pairs": len(eval_dataset),
            "seed": SEED,
        },
    )

    trainer = BiPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        # Same split twice: BiPOTrainer reads the injection sign off the name, so this is
        # one epoch's validation loss measured at +1 and at -1.
        eval_dataset={name: eval_dataset for name in EVAL_DATASETS},
        processing_class=tokenizer,
        behavior=args.behavior,
        layer=args.layer,
        name=args.id,
        callbacks=[saver],
    )

    trainer.train()

    if saver.best_epoch is None:
        raise SystemExit(
            "no epoch completed both evaluation passes, so no vectors were written -- "
            "check that eval_strategy fired at least once"
        )
    print(
        f"[{args.id}] best epoch {saver.best_epoch} (val_loss {saver.best_loss:.6f}); "
        f"vectors and report in {vec_dir}"
    )


if __name__ == "__main__":
    main()
