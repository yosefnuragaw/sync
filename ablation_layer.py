"""Per-layer ablation: find the layer each persona's steering vector works best at.

Samples `num_samples` rows per persona, builds a two-option forced choice (A = matching,
B = not_matching), and sweeps the persona's vector across layer 0 .. total_layer-1 one
layer at a time, scoring each by accuracy under the mean token log-probability of each
option. Labels are balanced over the sample, so accuracy is bidirectional.

Outputs, under `{output_dir}`:
    layer.csv      - the `top_k` best layers per persona (rank, layer, accuracy)
    layer_all.csv  - the full per-layer sweep, for plotting

Usage:
    python ablation_layer.py --config billy/configs/layer-ablation/llama3_1/persona-1.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from models.wrapper import CAABlockWrapper
from models.vectors import (
    ScaleFactors,
    print_scale_ranges,
    reference_scale_factors,
    unit_scale_factors,
    rescale,
)
from utils.common import batch_logps, set_seed
from utils.dataset import MultipleOptionDataset
from utils.prompts import SYSTEM_PROMPT

VEC_SCALES = ("raw", "reference", "unit")
SEED = 42
REPO_ROOT = Path(__file__).resolve().parent

OPT = ["A", "B"]

PERSONA_VEC_MAP: dict[str, list[str]] = {
    "persona-1": ["power-seeking", "wealth-seeking", "corrigible-less", "coordinate-other"],
    "single-ablation": ["power-seeking", "wealth-seeking"],
    "persona-2": ["machiavellianism", "narcissism", "psychopathy"],
    "persona-3": ["evil", "hallucinating", "sycophantic"],
    # persona-3 pair subsets; A = evil, B = hallucinating, C = sycophantic.
    "persona-3-ab": ["evil", "hallucinating"],
    "persona-3-ac": ["evil", "sycophantic"],
    "persona-3-bc": ["hallucinating", "sycophantic"],
}


@dataclass
class ScriptArguments:
    id: str = field(default="layer-ablation", metadata={"help": "Run id"})
    model_name_or_path: str = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint"},
    )
    behavior: str = field(
        default="persona-1",
        metadata={"help": "A persona group from PERSONA_VEC_MAP, or a single behaviour name"},
    )
    total_layer: int = field(default=32, metadata={"help": "LLM total number of layers"})
    layer: List[int] = field(
        default_factory=list,
        metadata={"help": "Layers to sweep; empty means 0 .. total_layer-1"},
    )
    vec_dir: Optional[List[str]] = field(
        default=None,
        metadata={"help": "One vector directory per persona, in the same order as the expanded behaviour list"},
    )
    multiplier: float = field(default=1.0, metadata={"help": "Steering magnitude; the sign is set per item by the label"})
    num_samples: int = field(default=100, metadata={"help": "Rows sampled from each persona's train.csv"})
    top_k: int = field(default=5, metadata={"help": "How many best layers to keep per persona"})
    output_dir: str = field(default="billy/ablation", metadata={"help": "Directory where layer.csv is written"})
    vec_scale: str = field(
        default="raw",
        metadata={
            "help": "How the stored vectors are scaled before the sweep: 'raw' (their own norms), "
            "'reference' (each one rescaled per layer onto the norm of its ref_vec_dir twin, "
            "keeping its direction but taking the reference magnitude), or 'unit' (every vector "
            "normalised per layer, so the multiplier IS the injected displacement)"
        },
    )
    ref_vec_dir: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Reference vector dirs for vec_scale 'reference': one per vec_dir entry, same "
            "order; normally the billy/vector/... twin of each vector being swept"
        },
    )
    apply_type: str = field(default="billy", metadata={"help": "CAABlockWrapper apply_type"})
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "e.g. flash_attention_2; leave unset to use the transformers default"},
    )


def wrapped_layers(model: PreTrainedModel) -> Iterator[tuple[int, CAABlockWrapper]]:
    """Yield (index, wrapper) for every steering-wrapped layer."""
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, CAABlockWrapper):
            yield idx, layer


def init_model(
    model_name: str,
    apply_type: str,
    total_layer: int = 32,
    attn_implementation: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the model and wrap its first `total_layer` blocks with zeroed steering vectors."""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    available = len(model.model.layers)
    if not 0 < total_layer <= available:
        raise ValueError(f"total_layer must be in 1..{available} for {model_name}, got {total_layer}")

    hidden_dim = model.config.hidden_size
    for layer in range(total_layer):
        wrapper = CAABlockWrapper(
            model.model.layers[layer],
            hidden_dim=hidden_dim,
            # (1, D): dim 0 is the persona axis; this sweep injects one at a time.
            vec=torch.zeros(1, hidden_dim, dtype=model.dtype),
            apply_type=apply_type,
            layer_idx=layer,
        )
        wrapper.extract(False)  # inference mode
        model.model.layers[layer] = wrapper

    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def zero_all_vectors(model: PreTrainedModel) -> None:
    """Reset every wrapped layer to a no-op zero vector."""
    hidden_dim = model.config.hidden_size
    for _, layer in wrapped_layers(model):
        layer.set_vec(torch.zeros(1, hidden_dim, dtype=model.dtype, device=model.device))
        layer.set_multiplier(1.0)


def load_layer_vector(
    vec_dir: str,
    layer: int,
    model: PreTrainedModel,
    factor: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Load this layer's positive steering vector as (1, D), or None when it is missing.

    Only the positive vector is read: extraction writes `neg = -pos`, so the negative
    direction is reached by flipping the multiplier's sign.

    `factor` is vec_scale 'reference''s per-layer rescale, from reference_scale_factors();
    None under 'raw', which sweeps at the norms the vectors were written with.
    """
    pos_path = os.path.join(vec_dir, f"pos_layer_{layer}.pt")
    if not os.path.exists(pos_path):
        print(f"[warning] missing vector: {pos_path}")
        return None
    vec = torch.load(pos_path, map_location=model.device)
    if factor is not None:
        vec = rescale(vec, factor)
    return vec.reshape(1, -1).to(dtype=model.dtype, device=model.device)


def collate_single(batch: list[dict]) -> dict:
    """Collate one MultipleOptionDataset item; options stay a list of (1, L) tensors.

    Batching is fixed at one item because the two options have different lengths and
    the scoring slice below is taken against a single question length.
    """
    item = batch[0]
    return {
        "question_length": item["question_length"],
        "input_ids": [ids.unsqueeze(0) for ids in item["input_ids"]],
        "attention_mask": [mask.unsqueeze(0) for mask in item["attention_mask"]],
        "label": [item["label"]],
    }


def build_loader(
    tokenizer: PreTrainedTokenizerBase,
    behavior: str,
    num_samples: int,
    seed: int,
) -> DataLoader:
    """Sample `num_samples` train rows into an A/B forced-choice loader with balanced labels."""
    path = REPO_ROOT / "data" / behavior / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    dataset = load_dataset("csv", data_files=str(path), split="train")
    rows = [
        row
        for row in dataset
        if row["question"] is not None and row["matching"] is not None and row["not_matching"] is not None
    ]
    if not rows:
        raise ValueError(f"No usable rows in {path}")

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(num_samples, len(rows)))
    if len(sampled) < num_samples:
        print(f"[warning] {behavior}: only {len(sampled)} usable rows, asked for {num_samples}")

    # Half the items test the positive direction (expect A, the matching answer), half
    # the negative one (expect B) — see the module docstring.
    n_pos = len(sampled) // 2
    labels = [OPT[0]] * n_pos + [OPT[1]] * (len(sampled) - n_pos)
    rng.shuffle(labels)

    questions, options = [], []
    for row in sampled:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["question"])},
        ]
        questions.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        options.append([str(row["matching"]), str(row["not_matching"])])

    return DataLoader(
        MultipleOptionDataset(tokenizer, prompts=options, questions=questions, labels=labels),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
    )


def eval_accuracy(
    model: PreTrainedModel,
    loader: DataLoader,
    multiplier: float,
    layers: List[int],
    verbose: bool = False,
) -> float:
    """Fraction of items where the steered model's preferred option matches the label."""
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Evaluating", ncols=100) if verbose else loader

    for batch in pbar:
        label = batch["label"][0]
        q_len = batch["question_length"]

        for layer in layers:
            block = model.model.layers[layer]
            if isinstance(block, CAABlockWrapper):
                block.set_multiplier(multiplier if label == OPT[0] else -multiplier)

        avg_logp = []
        for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

            with torch.no_grad():
                # float32 before the log_softmax: bf16 is too coarse for the mean
                # log-prob difference the two options are compared by.
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
                logps, _ = batch_logps(logits, input_ids)

                # logps[i] scores token i+1, so the option starts at q_len - 1.
                sliced = logps[0, q_len - 1:]
                avg_logp.append(sliced.mean().item())

        pred = OPT[avg_logp.index(max(avg_logp))]

        total += 1
        if pred == label:
            correct += 1

        if verbose:
            pbar.set_description(
                f"Evaluating [Multiplier:] {multiplier} [Accuracy:] {correct / total:.4f}"
            )

    return correct / total


def sweep_persona(
    model: PreTrainedModel,
    loader: DataLoader,
    persona: str,
    vec_dir: str,
    layers: List[int],
    multiplier: float,
    vec_idx: int = 0,
    factors: Optional[ScaleFactors] = None,
) -> list[dict]:
    """Score every layer in `layers` with this persona's vector, one layer at a time.

    Under vec_scale 'reference' the vector is rescaled onto its twin's norm first, so the
    layers are ranked at a comparable displacement.
    """
    records = []
    for layer in tqdm(layers, desc=f"{persona}: layers", ncols=100):
        factor = None if factors is None else factors[(vec_idx, layer, "pos")]
        vec = load_layer_vector(vec_dir, layer, model, factor)
        if vec is None:
            continue

        block = model.model.layers[layer]
        if not isinstance(block, CAABlockWrapper):
            print(f"[warning] layer {layer} is not wrapped, skipping")
            continue

        block.set_vec(vec)
        accuracy = eval_accuracy(model, loader, multiplier, [layer], verbose=False)
        zero_all_vectors(model)

        print(f"{persona} | layer {layer:>3} | accuracy {accuracy:.4f}")
        records.append({"persona": persona, "layer": layer, "accuracy": accuracy})
    return records


def top_layers(records: list[dict], top_k: int) -> list[dict]:
    """Rank one persona's layers by accuracy, ties broken by the earlier layer."""
    ranked = sorted(records, key=lambda r: (-r["accuracy"], r["layer"]))[:top_k]
    return [{**row, "rank": rank} for rank, row in enumerate(ranked, start=1)]


def load_layer_scores(csv_path: str) -> dict[str, list[tuple[int, float]]]:
    """Read an ablation csv into {persona: [(layer, accuracy), ...]}, best layer first.

    Accepts either layer.csv (the top_k rows per persona) or layer_all.csv (every layer):
    rows are re-sorted by accuracy with ties broken by the earlier layer, so the `rank`
    column is not required.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ablation results not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = {"persona", "layer", "accuracy"} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing column(s): {sorted(missing)}")

    scores: dict[str, list[tuple[int, float]]] = {}
    for persona, group in frame.groupby("persona", sort=False):
        ranked = group.sort_values(["accuracy", "layer"], ascending=[False, True])
        scores[str(persona)] = [(int(row.layer), float(row.accuracy)) for row in ranked.itertuples()]
    return scores


def assign_layers(
    personas: List[str],
    scores: dict[str, list[tuple[int, float]]],
    allowed_layers: Optional[List[int]] = None,
) -> dict[int, tuple[int, float]]:
    """Give each persona one exclusive layer; returns {layer: (persona index, accuracy)}.

    Candidates are consumed best-accuracy-first across all personas at once, so a persona
    that scores higher picks earlier. Equal cases go to the lower persona index — its
    position in `personas`, i.e. in vec_dir — and the persona that loses the tie falls
    through to its next-best free layer. Layers no persona claims stay unassigned.
    """
    allowed = None if allowed_layers is None else set(allowed_layers)

    candidates: list[tuple[float, int, int]] = []
    for index, persona in enumerate(personas):
        if persona not in scores:
            raise ValueError(f"No ablation rows for persona {persona!r}; found {sorted(scores)}")
        for layer, accuracy in scores[persona]:
            if allowed is None or layer in allowed:
                candidates.append((-accuracy, index, layer))

    # (accuracy desc, persona index asc, layer asc)
    candidates.sort()

    assignment: dict[int, tuple[int, float]] = {}
    claimed: set[int] = set()
    for neg_accuracy, index, layer in candidates:
        if index in claimed or layer in assignment:
            continue
        assignment[layer] = (index, -neg_accuracy)
        claimed.add(index)

    unassigned = [persona for index, persona in enumerate(personas) if index not in claimed]
    if unassigned:
        raise ValueError(
            f"No free layer left for {unassigned}: every candidate was taken by an earlier "
            f"persona. Re-run with a larger top_k, or point at layer_all.csv instead."
        )
    return assignment


def resolve_behaviors(behavior: str) -> List[str]:
    """Expand a persona group into its member behaviours, or pass a single one through."""
    return PERSONA_VEC_MAP.get(behavior, [behavior])


def resolve_paths(args: ScriptArguments) -> tuple[List[str], List[str], str]:
    """Validate the vector settings and return the paths the sweep reads and writes.

    Every path is used as the config gives it, relative to the repository root.
    """
    if not args.vec_dir:
        raise ValueError("vec_dir must be set in the config file")
    vec_dirs = list(args.vec_dir)
    ref_vec_dirs = list(args.ref_vec_dir or [])

    if args.vec_scale not in VEC_SCALES:
        raise ValueError(f"vec_scale must be one of {VEC_SCALES}, got {args.vec_scale!r}")
    if args.vec_scale == "reference" and len(ref_vec_dirs) != len(vec_dirs):
        raise ValueError(
            "vec_scale 'reference' needs one ref_vec_dir per vec_dir, in the same order; got "
            f"{len(ref_vec_dirs)} reference dir(s) for {len(vec_dirs)} vector dir(s)"
        )

    return vec_dirs, ref_vec_dirs, args.output_dir


def save(output_dir: str, file_name: str, df: pd.DataFrame) -> None:
    """Write df to output_dir/file_name, creating any intermediate directories."""
    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main(args: ScriptArguments) -> None:
    vec_dirs, ref_vec_dirs, output_dir = resolve_paths(args)
    behaviors = resolve_behaviors(args.behavior)
    if len(vec_dirs) != len(behaviors):
        raise ValueError(
            f"vec_dir has {len(vec_dirs)} entries but behaviour {args.behavior!r} expands to "
            f"{len(behaviors)}: {behaviors}"
        )

    layers = args.layer or list(range(args.total_layer))
    out_of_range = [l for l in layers if not 0 <= l < args.total_layer]
    if out_of_range:
        raise ValueError(f"layer entries out of range 0..{args.total_layer - 1}: {out_of_range}")

    # Read and validate every .pt before the model loads, so a missing or zero-norm
    # reference fails in seconds.
    factors = None
    if args.vec_scale == "reference":
        factors = reference_scale_factors(vec_dirs, ref_vec_dirs, layers)
        print_scale_ranges(behaviors, ref_vec_dirs, layers, factors)
    elif args.vec_scale == "unit":
        factors = unit_scale_factors(vec_dirs, layers)
        print_scale_ranges(behaviors, [], layers, factors, target="unit norm")
    print(f"[ablation] vec_scale={args.vec_scale}, sweeping {len(layers)} layer(s) per behaviour")

    model, tokenizer = init_model(
        model_name=args.model_name_or_path,
        apply_type=args.apply_type,
        total_layer=args.total_layer,
        attn_implementation=args.attn_implementation,
    )
    zero_all_vectors(model)

    all_records: list[dict] = []
    best_records: list[dict] = []

    for vec_idx, (persona, vec_dir) in enumerate(zip(behaviors, vec_dirs)):
        loader = build_loader(
            tokenizer=tokenizer,
            behavior=persona,
            num_samples=args.num_samples,
            seed=SEED,
        )
        records = sweep_persona(
            model=model,
            loader=loader,
            persona=persona,
            vec_dir=vec_dir,
            layers=layers,
            multiplier=args.multiplier,
            vec_idx=vec_idx,
            factors=factors,
        )
        if not records:
            print(f"[warning] {persona}: no layer produced a score, skipping")
            continue

        all_records.extend(records)
        best_records.extend(top_layers(records, args.top_k))

    if not all_records:
        raise RuntimeError("No layers were scored; check vec_dir and total_layer")

    save(output_dir, "layer_all.csv", pd.DataFrame(all_records))
    save(
        output_dir,
        "layer.csv",
        pd.DataFrame(best_records)[["persona", "rank", "layer", "accuracy"]],
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> ScriptArguments:
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


if __name__ == "__main__":
    set_seed(seed=SEED)
    main(parse_args())
