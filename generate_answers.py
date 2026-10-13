"""Generate steered answers for a behaviour across a sweep of steering multipliers.

For each multiplier the wrapped layers are pointed at the positive (multiplier > 0) or
negative steering vector and the model answers the behaviour's `test_infer.csv`. Where
each persona is injected is set by `apply_type`: one averaged direction at one layer
(`billy`), one persona per layer from an offline sweep (`simultaneous`, `sync-static`),
or a layer re-chosen every decode step (`sync`).

Outputs:
    {answer_dir}/{behaviour}/{multiplier}.csv

Usage:
    python generate_answers.py --config billy/configs/persona-1/llama3_1/sync.yaml
    python generate_answers.py --config <cfg> --baseline   # no steering
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_layer import assign_layers, load_layer_scores  # noqa: E402
from models.vectors import (  # noqa: E402
    print_scale_ranges,
    reference_scale_factors,
    unit_scale_factors,
    rescale,
)
from models import wrapper as _wrapper  # noqa: E402  (module object: 'sync' sets
                                       #  BUDGET_K / BUDGET_LAYERS on it)
from models.wrapper import CAABlockWrapper, LEVEL_APPLY_TYPES  # noqa: E402
from utils.common import set_seed  # noqa: E402
from utils.dataset import PromptDataset  # noqa: E402
from utils.prompts import SYSTEM_PROMPT  # noqa: E402

SEED = 42
GENERATION_BATCH_SIZE = 128
VEC_SCALES = ("raw", "reference", "unit")
REPO_ROOT = Path(__file__).resolve().parent

# Behaviour groups that expand into several per-behaviour generation runs.
PERSONA_VEC_MAP: dict[str, list[str]] = {
    "persona-1": ["power-seeking", "wealth-seeking", "corrigible-less", "coordinate-other"],
    "single-ablation": ["power-seeking", "wealth-seeking"],
    "persona-2": ["machiavellianism", "narcissism", "psychopathy"],
    "persona-3": ["evil", "hallucinating", "sycophantic"],
    # persona-3 pair subsets. Members and vector order match persona-3's, so a pair run
    # reuses the persona-3 vectors unchanged.
    "persona-3-ab": ["evil", "hallucinating"],
    "persona-3-ac": ["evil", "sycophantic"],
    "persona-3-bc": ["hallucinating", "sycophantic"],
    # Attribute-subset sweeps. Same behaviours as the group they come from: which
    # attributes are STEERED lives in each config's vec_dir, so a run is still scored on
    # all of them. Separate keys so these land in {tree}/reasoning/cross{,-3}/.
    "cross": ["power-seeking", "wealth-seeking", "corrigible-less", "coordinate-other"],
    "cross-3": ["evil", "hallucinating", "sycophantic"],
}

# Behaviours whose test_infer.csv ships no A/B option columns; the matching label comes
# straight from the matching / not_matching columns instead.
BEHAVIOURS_WITHOUT_OPTIONS = set(PERSONA_VEC_MAP["persona-2"]) | set(PERSONA_VEC_MAP["persona-3"])


@dataclass
class ScriptArguments:
    id: str = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: str = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint"},
    )
    behavior: str = field(default="power-seeking", metadata={"help": "the behavior"})
    steered_only: bool = field(
        default=False,
        metadata={"help": "Generate (and score) ONLY the behaviours this run actually "
                          "steers, i.e. those named in vec_dir, instead of the full list "
                          "`behavior` expands to. Off by default: the attribute-subset "
                          "sweeps ('cross', 'cross-3') deliberately score every behaviour "
                          "so an attribute ABSENT from vec_dir that still moves gives a "
                          "pure cross-attribute transfer reading. Turn it on to spend the "
                          "GPU only on the cells where something was injected -- a "
                          "singleton then writes one prompt set, a pair two. `behavior` "
                          "is NOT changed by this, so the reasoning tree and the analysis "
                          "keys stay where they are; only the member list shrinks. "
                          "Ignored under --baseline, which steers nothing."},
    )
    layer: List[int] = field(
        default_factory=lambda: list(range(32)),
        metadata={"help": "Layers to apply steering"},
    )
    total_layer: int = field(default=32, metadata={"help": "LLM total number of layers"})
    multipliers: List[float] = field(
        default_factory=lambda: [-2.0, -1.5, -1.0, -0.5, 2.0, 1.5, 1.0, 0.5],
        metadata={"help": "Steering multipliers to evaluate"},
    )
    level_baseline: str = field(
        default="running",
        metadata={"help": "apply_type 'sync' only. Default 'running': "
                          "(estimate the per-behaviour cosine offset online across "
                          "prompts -- no calibration file, but order-dependent since "
                          "the pipeline generates one behaviour at a time), or a .pt "
                          "(total_layer, N) tensor of mean cos(h, v_p) per layer per "
                          "behaviour, columns in vec_dir order. Subtracted before the "
                          "softmax so the gate is not biased toward whichever vector "
                          "has the largest natural cosine. Calibrate it on a BALANCED "
                          "prompt set (all behaviours) or at multiplier 0, otherwise it "
                          "absorbs the signal it is meant to expose."},
    )
    level_eps: float = field(
        default=1e-3,
        metadata={"help": "floor on the across-behaviour std when standardising the "
                          "level. Too small turns a near-tie into a confident guess."},
    )
    B: int = field(
        default=1,
        metadata={"help": "apply_type 'sync' only: how many layers each behaviour "
                          "may claim. 1 matches 'simultaneous' "
                          "exactly (one layer per persona); raising it trades selectivity "
                          "for coverage. Collisions FALL THROUGH to the next-best free "
                          "layer rather than dropping the loser, so the realised count is "
                          "exactly budget x n_behaviours while layers remain -- confirm "
                          "with models.wrapper.BUDGET_STATS, which should report no drops. "
                          "For SYNC-11 this budget is also what the PROMPT PASS spends."},
    )
    rho: float = field(
        default=1.0,
        metadata={"help": "apply_type 'sync' and 'sync-static' only: shape of the "
                          "prefill ramp r_t = ((t+1)/T)**p. 1.0 linear, >1 holds back then "
                          "rises late, <1 reaches full magnitude early. The last prompt "
                          "token is always full magnitude."},
    )
    warmup_reverse: bool = field(
        default=False,
        metadata={"help": "apply_type 'sync' and 'sync-static' only: reverse the "
                          "prefill ramp so it "
                          "starts at full magnitude and decays to ~0 at the LAST prompt "
                          "position. The forward ramp ends at r=1 and so never relieves "
                          "the hand-off into generation, which is where the first-token "
                          "breakage comes from."},
    )
    vec_dir: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Directories where .pt vectors are saved, one per behaviour vector"},
    )
    vec_scale: str = field(
        default="raw",
        metadata={
            "help": "How the stored vectors are scaled before steering: 'raw' (their own norms), "
            "'reference' (each one rescaled per layer onto the norm of its ref_vec_dir twin, "
            "keeping its direction but taking the reference magnitude), or 'unit' (every vector "
            "normalised per layer, so the multiplier IS the injected displacement and one "
            "multiplier grid means the same thing across behaviours, layers and personas)"
        },
    )
    ref_vec_dir: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Reference vector dirs for vec_scale 'reference': one per vec_dir entry, same "
            "order; normally the billy/vector/... twin of each vector being scaled"
        },
    )
    answer_dir: str = field(
        default="generation_results",
        metadata={"help": "Directory where CSVs will be saved"},
    )
    apply_type: str = field(
        default="billy",
        metadata={"help": "CAABlockWrapper apply_type: billy, sync, simultaneous or sync-static"},
    )
    ablation_csv: Optional[str] = field(
        default=None,
        metadata={
            "help": "layer.csv from ablation_layer.py, used by apply_type 'simultaneous' to pin "
            "each persona to a layer; defaults to billy/ablation/{behavior}/{model}/layer.csv"
        },
    )
    max_new_tokens: int = field(default=200, metadata={"help": "Max new generation tokens"})
    K: int = field(default=4, metadata={"help": "Lower bound sensitivity"})
    temp: float = field(
        default=1.0,
        metadata={"help": "INERT: kept so configs that set it still load; no surviving apply_type takes a softmax"},
    )


def wrapped_layers(model: PreTrainedModel) -> Iterator[tuple[int, CAABlockWrapper]]:
    """Yield (index, wrapper) for every steering-wrapped layer."""
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, CAABlockWrapper):
            yield idx, layer


def load_layer_vectors(
    vec_dirs: List[str],
    layer: int,
    layers: List[int],
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
    baseline: bool,
    factors: Optional[dict[tuple[int, int, str], float]] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Stack this layer's (pos, neg) steering vectors across vec_dirs into (N, D) tensors.

    Layers outside `layers`, and any missing vector file, contribute zeros so the
    stack keeps a consistent shape. Returns None when there are no vector dirs at
    all, which is the --baseline case.

    `factors` is vec_scale 'reference''s rescaling, from reference_scale_factors(); None
    under 'raw', which loads the vectors at the norms they were written with.
    """
    if not vec_dirs:
        return None

    pos_vecs, neg_vecs = [], []
    for idx, vec_dir in enumerate(vec_dirs):
        pos_path = os.path.join(vec_dir, f"pos_layer_{layer}.pt")
        neg_path = os.path.join(vec_dir, f"neg_layer_{layer}.pt")
        steer_this_layer = layer in layers

        if not baseline and steer_this_layer and os.path.exists(pos_path) and os.path.exists(neg_path):
            pos = torch.load(pos_path, map_location=device)
            neg = torch.load(neg_path, map_location=device)
            if factors is not None:
                pos = rescale(pos, factors[(idx, layer, "pos")])
                neg = rescale(neg, factors[(idx, layer, "neg")])
            pos_vecs.append(pos)
            neg_vecs.append(neg)
            continue

        if not baseline and steer_this_layer:
            print(f"Missing vectors: {pos_path} or {neg_path}")
        pos_vecs.append(torch.zeros(hidden_dim, dtype=dtype, device=device))
        neg_vecs.append(torch.zeros(hidden_dim, dtype=dtype, device=device))

    return torch.stack(pos_vecs, dim=0), torch.stack(neg_vecs, dim=0)


def init_model(
    model_name: str,
    vec_dirs: List[str],
    layers: List[int],
    apply_type: str,
    vec_names: Optional[List[str]],
    K: int,
    total_layer: int = 32,
    baseline: bool = False,
    temp: float = 1.0,
    layer_assignment: Optional[dict[int, int]] = None,
    vec_scale: str = "raw",
    ref_vec_dirs: Optional[List[str]] = None,
    level_baseline: Optional[str] = None,
    level_eps: float = 1e-3,
    B: int = 1,
    rho: float = 1.0,
    warmup_reverse: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
    """Load the model, wrap its first `total_layer` blocks, and preload steering vectors.

    Returns the model, tokenizer, and a per-layer list of (pos, neg) vector stacks.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    available = len(model.model.layers)
    if not 0 < total_layer <= available:
        raise ValueError(f"total_layer must be in 1..{available} for {model_name}, got {total_layer}")

    hidden_dim = model.config.hidden_size
    names = vec_names or [f"vec-{i}" for i in range(len(vec_dirs))]
    space: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []

    # (total_layer, N) per-layer, per-behaviour cosine baseline for the level gate. It
    # de-biases online by default; a .pt path supplies a fixed calibration instead.
    if apply_type in _wrapper.BUDGET_APPLY_TYPES:
        _wrapper.BUDGET_K = B
        _wrapper.BUDGET_LAYERS.clear()
        _wrapper.reset_budget_state()
        _wrapper.BUDGET_STATS.update(steps=0, placed=0, dropped=0)
        print(f"[budget] each behaviour may claim {B} layer(s); at most "
              f"{B * len(vec_dirs)} of {len(layers)} will inject per step")

    lvl_base = None
    if apply_type in LEVEL_APPLY_TYPES and level_baseline in (None, "", "running"):
        lvl_base = "running"
        print("[level] de-biasing online across prompts (no calibration file)")
    elif level_baseline and level_baseline != "running":
        if not os.path.exists(level_baseline):
            raise FileNotFoundError(
                f"level_baseline {level_baseline!r} not found. Omit it to de-bias "
                "online instead -- that is the default and needs no calibration run.")
        lvl_base = torch.load(level_baseline, map_location="cpu").float()
        if not isinstance(lvl_base, str) and lvl_base.shape != (total_layer, len(vec_dirs)):
            raise ValueError(
                f"level_baseline {level_baseline} has shape {tuple(lvl_base.shape)}, "
                f"expected ({total_layer}, {len(vec_dirs)}) -- one row per layer, one "
                "column per vec_dir in the same order")
        print(f"[level] baseline from {level_baseline}, per-behaviour means "
              f"{[round(float(x), 5) for x in lvl_base.mean(0)]}")
    factors = None
    if vec_dirs and not baseline:
        if vec_scale == "reference":
            factors = reference_scale_factors(vec_dirs, ref_vec_dirs or [], layers)
            print_scale_ranges(names, ref_vec_dirs or [], layers, factors)
        elif vec_scale == "unit":
            factors = unit_scale_factors(vec_dirs, layers)
            print_scale_ranges(names, [], layers, factors, target="unit norm")

    # `layer` defaults to every layer, so an omitted list would mark all 32 steered.
    # Baseline injects nothing.
    steered: set[int] = set() if baseline else set(layers)
    for layer in range(total_layer):
        space.append(
            load_layer_vectors(
                vec_dirs=vec_dirs,
                layer=layer,
                layers=layers,
                hidden_dim=hidden_dim,
                dtype=model.dtype,
                device=model.device,
                baseline=baseline,
                factors=factors,
            )
        )
        wrapper = CAABlockWrapper(
            model.model.layers[layer],
            hidden_dim=hidden_dim,
            # (N, D), one row per persona: the shape every apply_type branch expects.
            # The placeholder carries the persona axis too.
            vec=torch.zeros(max(len(vec_dirs), 1), hidden_dim, dtype=model.dtype),
            apply_type=apply_type,
            K=K,
            temp=temp,
            layer_idx=layer,
            dir_idx=None if layer_assignment is None else layer_assignment.get(layer),
            level_baseline=(lvl_base if isinstance(lvl_base, str)
                            else None if lvl_base is None else lvl_base[layer]),
            level_eps=level_eps,
            rho=rho,
            warmup_reverse=warmup_reverse,
        )
        wrapper.extract(False)  # inference mode
        wrapper.set_vec_names(names)
        wrapper.set_steer(layer in steered)
        model.model.layers[layer] = wrapper

    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, space


def read_dataset(
    behavior: str,
    tokenizer: PreTrainedTokenizerBase,
    multiplier: float = 0.0,
) -> dict[str, list]:
    """Build chat-templated prompts and the expected matching label for one behaviour.

    A negative multiplier steers toward the opposite behaviour, so the expected
    label is flipped to match.
    """
    path = REPO_ROOT / "data" / behavior / "test_infer.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    dataset = load_dataset("csv", data_files=str(path), split="train")

    results: dict[str, list] = {
        "prompts": [],
        "questions": [],
        "A": [],
        "B": [],
        "answers": [],
        "matching": [],
    }
    has_options = behavior not in BEHAVIOURS_WITHOUT_OPTIONS
    steering_inverted = multiplier < 0

    for idx, row in enumerate(dataset):
        question = row.get("question")
        if question is None:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(question)},
        ]
        results["prompts"].append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        results["questions"].append(question)

        if has_options:
            if "A" not in row or "B" not in row:
                raise ValueError(f"Expected column A and B in row {idx}")
            results["A"].append(row["A"])
            results["B"].append(row["B"])
            matching = row["matching"]
            if steering_inverted:
                matching = "B" if matching == "A" else "A"
            results["matching"].append(matching)
        else:
            results["A"].append("")
            results["B"].append("")
            results["matching"].append(row["not_matching"] if steering_inverted else row["matching"])

    return results


def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: dict[str, list],
    max_new_tokens: int = 200,
    batch_size: int = GENERATION_BATCH_SIZE,
) -> dict[str, list]:
    """Greedily generate an answer per prompt; returns the dataset with prompts swapped for answers."""
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    prompt_loader = PromptDataset(dataset["prompts"])

    answers = []
    for output in tqdm(
        generator(
            prompt_loader,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            batch_size=batch_size,
            return_full_text=False,
        ),
        total=len(prompt_loader),
        desc="Generating Answers",
    ):
        answers.append(output[0]["generated_text"].strip())

    result = {k: v for k, v in dataset.items() if k != "prompts"}
    result["answers"] = answers
    return result


def save(output_dir: str, file_name: str, df: pd.DataFrame) -> None:
    """Write df to output_dir/file_name, creating any intermediate directories."""
    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def point_layers_at_multiplier(
    model: PreTrainedModel,
    layers: List[int],
    space: list[Optional[tuple[torch.Tensor, torch.Tensor]]],
    multiplier: float,
) -> None:
    """Load the pos (multiplier > 0) or neg vector stack into each steered layer.

    The wrapper always receives a positive magnitude; the sign selects which
    stack it steers with.
    """
    pos_idx = 0 if multiplier > 0 else 1
    for idx in layers:
        layer = model.model.layers[idx]
        if not isinstance(layer, CAABlockWrapper):
            continue
        layer.set_multiplier(abs(multiplier))
        if space[idx] is not None:
            layer.set_vec(space[idx][pos_idx])


def default_ablation_csv(behavior: str, vec_dirs: List[str]) -> str:
    """Where ablation_layer.py writes this run's results: billy/ablation/{behavior}/{model}/layer.csv.

    The model folder is read back off the vector paths (.../{persona}/{model}/all), so it
    matches whichever model the vectors belong to.
    """
    model_dir = Path(vec_dirs[0]).parts[-2]
    return os.path.join("billy", "ablation", behavior, model_dir, "layer.csv")


def resolve_paths(args: ScriptArguments, baseline: bool = False) -> tuple[List[str], List[str], str, str]:
    """Validate the vector settings and settle the layer-assignment csv.

    Every path is used as the config gives it: relative to the repository root, which is
    where all four entry points are run from.
    """
    vec_dirs = list(args.vec_dir or [])
    ref_vec_dirs = list(args.ref_vec_dir or [])

    if args.vec_scale not in VEC_SCALES:
        raise ValueError(f"vec_scale must be one of {VEC_SCALES}, got {args.vec_scale!r}")
    if args.vec_scale == "reference":
        if len(ref_vec_dirs) != len(vec_dirs):
            raise ValueError(
                "vec_scale 'reference' needs one ref_vec_dir per vec_dir, in the same order; got "
                f"{len(ref_vec_dirs)} reference dir(s) for {len(vec_dirs)} vector dir(s)"
            )

    if not vec_dirs and not baseline:
        # A baseline config carries no vec_dir: it injects nothing. Every steered run
        # still has to name its vectors.
        raise ValueError("vec_dir must be set in the config file")

    ablation_csv = args.ablation_csv or (default_ablation_csv(args.behavior, vec_dirs) if vec_dirs else "")

    return vec_dirs, ref_vec_dirs, args.answer_dir, ablation_csv


def build_layer_assignment(
    ablation_csv: str,
    vec_names: List[str],
    layers: List[int],
) -> dict[int, int]:
    """Map each layer to the persona index it hosts under apply_type 'simultaneous'.

    Every persona is pinned to the layer it scored best on in the ablation sweep, ties
    going to the lower persona index; every other layer is left unsteered.
    """
    scores = load_layer_scores(ablation_csv)
    assignment = assign_layers(vec_names, scores, allowed_layers=layers)

    print(f"[simultaneous] layer assignment from {ablation_csv}")
    for layer in sorted(assignment):
        index, accuracy = assignment[layer]
        print(f"  layer {layer:>3} -> {vec_names[index]} (persona {index}, accuracy {accuracy:.4f})")
    idle = [layer for layer in layers if layer not in assignment]
    print(f"  {len(idle)} of {len(layers)} steered layer(s) left idle: {idle}")

    return {layer: index for layer, (index, _) in assignment.items()}


def check_vectors(vec_dirs: List[str], layers: List[int], role: str = "vec_dir") -> None:
    """Fail before the model is loaded if a steered layer has no vector to apply.

    load_layer_vectors() substitutes zeros for a missing file and prints one line about
    it, so the run looks normal while steering with nothing. vec_scale 'unit' and
    'reference' raise on their own, but only from inside init_model(), with the whole
    model already in memory; on 'raw' nothing else catches it.
    """
    missing = [
        os.path.join(vec_dir, f"{sign}_layer_{layer}.pt")
        for vec_dir in vec_dirs
        for layer in layers
        for sign in ("pos", "neg")
        if not os.path.exists(os.path.join(vec_dir, f"{sign}_layer_{layer}.pt"))
    ]
    if not missing:
        return
    tree = Path(missing[0]).parts[0]
    writer = "bipo/train.py" if tree == "bipo" else "billy/extract_caa.py"
    raise SystemExit(
        f"{len(missing)} steering vector file(s) missing under {role}, first: {missing[0]}\n"
        f"run {writer} for each behaviour in {role} first"
    )


def check_layer_assignment_source(ablation_csv: str, vec_dirs: List[str],
                                  explicit: bool) -> None:
    """Fail unless the layer sweep was ranked on the same vectors this run injects.

    'simultaneous' and 'sync-static' read their decode hosts out of `ablation_csv`, so
    the assignment is the intervention. default_ablation_csv() falls back to
    billy/ablation/..., which is right for a billy run and wrong for any other tree. The tree
    is read off vec_dirs and must match the sweep's; a mismatch is an error, never a
    warning, because the run it produces looks normal.
    """
    vec_tree = Path(vec_dirs[0]).parts[0]
    csv_tree = Path(ablation_csv).parts[0] if ablation_csv else ""
    if csv_tree != vec_tree:
        hint = ("" if explicit else
                " `ablation_csv` is unset, so this is the billy/ablation default.")
        raise SystemExit(
            f"layer assignment comes from {ablation_csv!r} ({csv_tree or 'nowhere'}) but the "
            f"vectors come from {vec_tree}/.{hint}\n"
            f"Set `ablation_csv` to the {vec_tree}/ablation/... layer.csv that "
            f"ablation_layer.py writes for these vectors."
        )
    if not os.path.exists(ablation_csv):
        raise SystemExit(
            f"ablation_csv not found: {ablation_csv}\n"
            "run ablation_layer.py with the matching layer-ablation config first"
        )


def resolve_behaviors(behavior: str) -> List[str]:
    """Expand a persona group into its member behaviours, or pass a single one through."""
    return PERSONA_VEC_MAP.get(behavior, [behavior])


def main(baseline: bool, args: ScriptArguments) -> None:
    vec_dirs, ref_vec_dirs, output_dir, ablation_csv = resolve_paths(args, baseline)
    vec_names = [Path(d).parts[-3] for d in vec_dirs]

    # Pre-flight, before the model load, so a bad path costs seconds not minutes.
    if not baseline:
        check_vectors(vec_dirs, args.layer)
        if args.vec_scale == "reference":
            check_vectors(ref_vec_dirs, args.layer, role="ref_vec_dir")
        for idx, vec_dir in enumerate(vec_dirs):
            ref = f"  -> norms from {ref_vec_dirs[idx]}" if args.vec_scale == "reference" else ""
            print(f"[vectors]   {vec_dir}{ref}")
        print(f"[vectors] vec_scale={args.vec_scale}")
    print(f"[answers] -> {output_dir}")

    # Baseline needs no assignment; every dir_idx stays None.
    layer_assignment = None
    if args.apply_type in ("simultaneous", "sync-static") and not baseline:
        # 'sync-static' reads the same assignment: it is 'simultaneous' during decode.
        check_layer_assignment_source(ablation_csv, vec_dirs, bool(args.ablation_csv))
        layer_assignment = build_layer_assignment(ablation_csv, vec_names, args.layer)

    model, tokenizer, space = init_model(
        model_name=args.model_name_or_path,
        vec_dirs=vec_dirs,
        layers=args.layer,
        apply_type=args.apply_type,
        vec_names=vec_names,
        K=args.K,
        total_layer=args.total_layer,
        baseline=baseline,
        temp=args.temp,
        layer_assignment=layer_assignment,
        vec_scale=args.vec_scale,
        level_baseline=args.level_baseline,
        level_eps=args.level_eps,
        B=args.B,
        rho=args.rho,
        warmup_reverse=args.warmup_reverse,
        ref_vec_dirs=ref_vec_dirs,
    )

    behaviors = resolve_behaviors(args.behavior)
    if args.steered_only and not baseline:
        steered = set(vec_names)
        kept = [b for b in behaviors if b in steered]
        if not kept:
            raise ValueError(
                f"steered_only left nothing to generate: behavior '{args.behavior}' "
                f"expands to {behaviors}, vec_dir steers {sorted(steered)}. The two "
                f"must overlap -- check that vec_dir points at this persona's vectors."
            )
        dropped = [b for b in behaviors if b not in steered]
        print(f"[steered_only] generating {kept}; skipping {dropped} "
              f"(not in vec_dir, so nothing would be injected on those prompts)")
        behaviors = kept

    for behav in behaviors:
        if baseline:
            multipliers: List[Optional[float]] = [None]
        else:
            multipliers = list(args.multipliers)

        for multiplier in multipliers:
            if multiplier is None:
                dataset = read_dataset(behavior=behav, tokenizer=tokenizer)
                file_name = f"{behav}/baseline.csv"
            else:
                point_layers_at_multiplier(model, args.layer, space, multiplier)
                dataset = read_dataset(behavior=behav, tokenizer=tokenizer, multiplier=multiplier)
                file_name = f"{behav}/{multiplier}.csv"

            answered = generate_answers(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                max_new_tokens=args.max_new_tokens,
            )
            save(output_dir, file_name, pd.DataFrame(answered))


def parse_args() -> tuple[bool, ScriptArguments]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to a .yaml or .json config")
    parser.add_argument("--baseline", action="store_true", help="Generate without steering")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warning] ignoring unrecognised arguments: {unknown}")

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    return args.baseline, script_args


if __name__ == "__main__":
    set_seed(seed=SEED)
    main(*parse_args())
