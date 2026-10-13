"""Per-layer scaling of stored steering vectors, for the generation and ablation runs.

Three settings of `vec_scale`:

    raw        inject each vector at the norm extraction gave it
    unit       rescale to norm 1, so the multiplier IS the injected displacement
    reference  keep the direction, take the per-layer magnitude from `ref_vec_dir`

Switching to 'unit' changes what every multiplier does, so it means re-running the sweep.
Nothing here touches a model or a config; it reads .pt files and returns numbers.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

# (vec_dir index, layer, "pos"|"neg") -> multiplicative factor
ScaleFactors = Dict[Tuple[int, int, str], float]


def unit_scale_factors(vec_dirs: List[str], layers: List[int]) -> ScaleFactors:
    """Per (vector, layer, sign) factor that takes every stored vector to unit norm."""
    factors: ScaleFactors = {}
    for idx, vec_dir in enumerate(vec_dirs):
        for layer in layers:
            for sign in ("pos", "neg"):
                path = os.path.join(vec_dir, f"{sign}_layer_{layer}.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"vec_scale 'unit' needs {path} to normalise layer {layer}")
                norm = float(torch.load(path, map_location="cpu").float().norm())
                if norm == 0.0:
                    raise ValueError(
                        f"layer {layer} {sign}: zero norm in {vec_dir} -- no direction to keep")
                factors[(idx, layer, sign)] = 1.0 / norm
    return factors


def reference_scale_factors(
    vec_dirs: List[str],
    ref_vec_dirs: List[str],
    layers: List[int],
) -> ScaleFactors:
    """Per (vector, layer, sign) factor that rescales a stored vector onto its reference norm.

    Keeps the direction in `vec_dirs` and takes the magnitude from `ref_vec_dirs`, so two
    vector sets displace the residual stream equally at the same multiplier. Every file is
    read up front: a missing or zero-norm vector would otherwise pass as a mis-scaled run.
    """
    factors: ScaleFactors = {}
    for idx, (vec_dir, ref_dir) in enumerate(zip(vec_dirs, ref_vec_dirs)):
        for layer in layers:
            for sign in ("pos", "neg"):
                name = f"{sign}_layer_{layer}.pt"
                vec_path = os.path.join(vec_dir, name)
                ref_path = os.path.join(ref_dir, name)
                for path in (vec_path, ref_path):
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"vec_scale 'reference' needs {path} to scale layer {layer} onto the "
                            "reference magnitude"
                        )
                norm = float(torch.load(vec_path, map_location="cpu").float().norm())
                target = float(torch.load(ref_path, map_location="cpu").float().norm())
                if norm == 0.0 or target == 0.0:
                    raise ValueError(
                        f"layer {layer} {sign}: norm {norm:.4g} in {vec_dir}, reference norm "
                        f"{target:.4g} in {ref_dir} -- a zero norm leaves no direction to keep or "
                        "no magnitude to match; steering would be a no-op"
                    )
                factors[(idx, layer, sign)] = target / norm
    return factors


def rescale(vec: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale a stored vector, in fp32 so a bf16 vector keeps its direction through the multiply."""
    return (vec.float() * factor).to(vec.dtype)


def print_scale_ranges(
    names: List[str],
    ref_vec_dirs: List[str],
    layers: List[int],
    factors: ScaleFactors,
    target: str = "",
) -> None:
    """Print the applied factor span per vector, logged by both scripts before loading."""
    # 'unit' has no reference dirs; name the target from `target` instead.
    dirs = ref_vec_dirs or [target] * len(names)
    for idx, ref_dir in enumerate(dirs):
        scales = [factors[(idx, layer, "pos")] for layer in layers]
        lo, hi = min(scales), max(scales)
        print(
            f"[scale] {names[idx]}: x{lo:.3g} (layer {layers[scales.index(lo)]}) .. "
            f"x{hi:.3g} (layer {layers[scales.index(hi)]}) onto {ref_dir}"
        )
