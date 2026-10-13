"""Extract CAA (Contrastive Activation Addition) steering vectors.

Runs the paired dataset twice per wrapped layer, once serving the `chosen` response and
once the `rejected` one, and writes the mean activation difference.

Outputs, under `{vec_dir}`:
    pos_layer_{i}.pt / neg_layer_{i}.pt

Usage:
    python billy/extract_caa.py --config billy/configs/extract/llama3_1/power.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wrapper import CAABlockWrapper  # noqa: E402
from utils.common import get_data, set_seed  # noqa: E402
from utils.dataset import PairedCAADataset  # noqa: E402

SEED = 42
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
EXTRACTION_MULTIPLIER = 1.0

DIRECTION_MODES: dict[int, str] = {1: "chosen", -1: "rejected"}


@dataclass
class ScriptArguments:
    id: str = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: str = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    total_layer: int = field(default=32, metadata={"help": "Number of layers to wrap and extract from"})
    behavior: str = field(default="power-seeking", metadata={"help": "the behavior"})
    vec_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory where .pt vectors are saved"},
    )
    apply_type: str = field(
        default="billy",
        metadata={"help": "CAABlockWrapper apply_type; irrelevant during extraction but must be valid"},
    )


def build_collate_fn(tokenizer: PreTrainedTokenizerBase) -> Callable[[list[dict]], dict[str, torch.Tensor]]:
    """Tokenise prompt+response pairs and mark which positions are response tokens."""

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        texts = [item["prompt"] + item["response"] for item in batch]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

        # response_mask is built and passed, but nothing consumes it: CAABlockWrapper
        # averages over all positions. Kept so the intended masking is not lost.
        response_mask = torch.zeros_like(enc["attention_mask"])
        for i, prompt in enumerate(prompts):
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"])
            seq_len = int(enc["attention_mask"][i].sum().item())
            if prompt_len < seq_len:
                response_mask[i, prompt_len:seq_len] = 1

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "response_mask": response_mask,
        }

    return collate_fn


def build_dataloader(
    behavior: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    data = get_data(tokenizer=tokenizer, behavior=behavior, generation_prompt=False)
    return DataLoader(
        PairedCAADataset(data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer),
    )


def wrapped_layers(model: PreTrainedModel) -> Iterator[tuple[int, CAABlockWrapper]]:
    """Yield (index, wrapper) for every layer that has been wrapped for extraction."""
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, CAABlockWrapper):
            yield idx, layer


def load_model_and_tokenizer(
    model_name: str,
    apply_type: str,
    total_layer: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the model, wrap its first `total_layer` blocks, and put it in extraction mode."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    available = len(model.model.layers)
    if not 0 < total_layer <= available:
        raise ValueError(f"total_layer must be in 1..{available} for {model_name}, got {total_layer}")

    hidden_dim = model.config.hidden_size
    for idx in range(total_layer):
        wrapper = CAABlockWrapper(
            model.model.layers[idx],
            hidden_dim=hidden_dim,
            vec=torch.zeros(hidden_dim, dtype=model.dtype),
            apply_type=apply_type,
        )
        wrapper.extract(True)
        model.model.layers[idx] = wrapper

    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _run_direction(model: PreTrainedModel, loader: DataLoader, direction: int, multiplier: float) -> None:
    """Accumulate activations for one contrast direction across the whole loader."""
    loader.dataset.set_mode(DIRECTION_MODES[direction])
    for _, layer in wrapped_layers(model):
        layer.set_multiplier(direction * multiplier)

    for batch in loader:
        with torch.no_grad():
            model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                response_mask=batch["response_mask"].to(model.device),
            )


def save_vectors(model: PreTrainedModel, vec_dir: str, verbose: bool = False) -> None:
    """Drain each wrapper's buffers into pos/neg steering vectors on disk."""
    os.makedirs(vec_dir, exist_ok=True)
    for idx, layer in wrapped_layers(model):
        multiplier = layer.multiplier
        vec_pos, vec_neg = layer.extract_vec(clear=True)
        if verbose:
            print(f"Layer {idx}: pos={tuple(vec_pos.shape)}, neg={tuple(vec_neg.shape)}, multiplier={multiplier}")
        torch.save(vec_pos, os.path.join(vec_dir, f"pos_layer_{idx}.pt"))
        torch.save(vec_neg, os.path.join(vec_dir, f"neg_layer_{idx}.pt"))


def extract_caa(
    model: PreTrainedModel,
    loader: DataLoader,
    vec_dir: str,
    multiplier: float = EXTRACTION_MULTIPLIER,
    verbose: bool = False,
) -> None:
    directions = list(DIRECTION_MODES)
    progress = tqdm(directions, desc="Extracting", ncols=100) if verbose else directions
    for direction in progress:
        _run_direction(model, loader, direction, multiplier)
    save_vectors(model, vec_dir, verbose=verbose)


def resolve_vec_dir(vec_dir: Optional[str]) -> str:
    """Where the extracted vectors are written, as the config gives it."""
    if not vec_dir:
        raise ValueError("vec_dir must be set in the config file")
    return vec_dir


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
    set_seed(seed=SEED)
    script_args = parse_script_arguments()
    vec_dir = resolve_vec_dir(script_args.vec_dir)

    model, tokenizer = load_model_and_tokenizer(
        model_name=script_args.model_name_or_path,
        apply_type=script_args.apply_type,
        total_layer=script_args.total_layer,
    )
    loader = build_dataloader(behavior=script_args.behavior, tokenizer=tokenizer)

    extract_caa(
        model=model,
        loader=loader,
        vec_dir=vec_dir,
        multiplier=EXTRACTION_MULTIPLIER,
        verbose=True,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
