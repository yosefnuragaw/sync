# BiPO

Bi-directional Preference Optimisation steering vectors, wired into this repository's
shared generation pipeline. The four steering rules, the config reference and the
end-to-end commands are in the [root README](../README.md); this file covers only what is
specific to the BiPO tree.

Training, the training-time block wrapper and the BiPO configs live in this `bipo/`
directory. `bipo/trl/` is a vendored TRL holding `BiPOTrainer`, a `DPOTrainer` that steers
the policy while it optimises.

CAA reads its vector off the data: the mean activation difference between the matching and
non-matching continuations. BiPO **learns** it instead. Every wrapped layer carries a
trainable vector, the base weights stay frozen, and the DPO objective is optimised against
a frozen reference copy of the same model. The trick that makes one vector work in both
directions is that the trainer flips the injection sign from batch to batch and flips the
preference with it:

```
h  <-  h + m * v          m ~ {+1, -1} per batch
loss = DPO(chosen, rejected)      at m = +1
       DPO(rejected, chosen)      at m = -1
```

so `+v` steers toward the behaviour and `-v` away from it, which is exactly the
`pos_layer_{i}.pt` / `neg_layer_{i}.pt` pair the generation side expects.

## Training

One run per behaviour per model, from the repository root (the data loader reads
`./data`):

```bash
python bipo/train.py --config bipo/configs/train/llama3_1/power.yaml
```

`bipo/configs/train/<model>/` ships 10 configs per model; 7 of them have data in `data/`.

Artifacts land in `bipo/vector/<behaviour>/<model>/all/`:

| file | contents |
| --- | --- |
| `pos_layer_{i}.pt` | the learned vector `v` for layer i, bf16, best epoch only |
| `neg_layer_{i}.pt` | `-v`, the same vector's negative direction |
| `bipo_train_report.json` | per-epoch validation losses, which epoch was kept, per-layer norms |

### Which epoch is kept

The held-out split is scored twice per epoch, once with the vector added (`add`, +1) and
once with it subtracted (`sub`, -1), and an epoch is ranked by the **mean** of the two. A
vector that only improves in one direction is not what generation applies, since
generation reads `-v` straight off disk as the negative steering vector.

The vectors on disk are always the best epoch's: a worse epoch never overwrites them, and
`bipo_train_report.json` is rewritten every epoch, so a run that dies part-way still
leaves both its history and its best vectors behind.

`data/<behaviour>/test.csv` is the validation split wherever it exists, which is the four
persona-1 behaviours. The persona-3 behaviours (evil, hallucinating, sycophantic) ship
only `train.csv` and `test_infer.csv`, and `test_infer.csv` carries no chosen/rejected
pair to score a DPO loss on, so there `val_fraction` of `train.csv` is held out instead
(seeded, so reruns get the same split).

Tracking is entirely local: per-epoch losses go to stdout and to the report JSON, and
`report_to` is pinned to `"none"`.

## Generation

BiPO vectors are generated through the repository-root `generate_answers.py`, the same
entry point the CAA arms use. `bipo/train.py` writes them in the CAA
`pos_layer_{i}.pt` / `neg_layer_{i}.pt` layout, so nothing about the application differs
once they are on disk — only where they came from.

```bash
python generate_answers.py --config bipo/configs/persona-1/llama3_1/15-billy-ref.yaml
python generate_answers.py --config bipo/configs/persona-1/llama3_1/simultaneous.yaml
python generate_answers.py --config bipo/configs/persona-1/llama3_1/sync.yaml
```

There used to be a `bipo/generate_answers.py` pre-flight wrapper in front of it. It was
removed and its two checks moved into the root runner, where they now cover both trees
and all four apply types rather than BiPO's `billy` and `simultaneous` alone. Both run
before the model is loaded:

- `check_vectors` — every steered layer has a vector file on disk, and the same for
  `ref_vec_dir` under `vec_scale: reference`. Without it `load_layer_vectors` substitutes
  zeros and prints one line that is easy to miss in a 32-layer run. `vec_scale: unit` and
  `reference` do raise on their own, but only from inside `init_model`, with the whole
  model already in memory; on `vec_scale: raw` nothing else catches it.
- `check_layer_assignment_source` — the layer sweep must come from the same tree as the
  vectors. `default_ablation_csv` falls back to `billy/ablation/{behavior}/{model}/layer.csv`
  whenever `ablation_csv` is unset, which is right for a CAA run and silently wrong here:
  `simultaneous` and `sync-static` **are** their layer assignment, and CAA's ranking
  would pin the BiPO vectors to the layers CAA's directions scored best on. A
  `bipo/vector/...` run pointed at a `billy/ablation/...` sweep is an error, not a warning.

### simultaneous

```
at layer L:  h  <-  h + multiplier * v_{p(L)}      p(L) from `ablation_csv`
```

`layer` is the candidate window (all 32); the assignment comes from an `ablation_layer.py`
sweep, collisions go to the earlier `vec_dir` entry, and every unclaimed layer is left
unsteered.

```bash
# 1. rank the layers on BiPO's own vectors -> bipo/ablation/<group>/<model>/layer.csv
python ablation_layer.py        --config bipo/configs/layer-ablation/llama3_1/persona-3.yaml

# 2. steer, one behaviour per layer
python generate_answers.py      --config bipo/configs/persona-3/llama3_1/simultaneous.yaml
```

The ablation sweep sets `vec_scale: reference`. It scores one layer at a time at a fixed
multiplier and then ranks the layers against each other, so the per-layer norm spread in
the table below would otherwise be part of what is being ranked, with layer 0 judged under
a far larger perturbation than layer 31. Rescaling onto the CAA twin's per-layer norm
keeps the learned direction and puts every layer at a comparable displacement.

That makes **`simultaneous.yaml` the consistent arm**: the layer is used under the
same scaling it was chosen under, so only the direction and the assignment differ from the
CAA `simultaneous` run at a fixed multiplier. `all-simultaneous.yaml`, which exists
for persona-3 only, is the raw parity arm; its layers were ranked at reference scale but
are steered at BiPO's own.

## Magnitude: `vec_scale`

`raw` injects the vectors at the norms BiPO learned. `reference` rescales each one, per
layer, onto the norm of its `ref_vec_dir` twin, the CAA vector for the same behaviour,
keeping the learned direction and taking CAA's magnitude:

```
v_p  <-  v_p * ||caa_p(layer)|| / ||v_p(layer)||
```

Every committed BiPO run config carries `-ref` in its name and sets `vec_scale:
reference`, except persona-3's `all-simultaneous.yaml`.

This exists because the two methods do not produce comparable lengths. CAA's mean
differences grow with the residual stream (Llama layer 0 to 31: 0.009 to 1.37), while BiPO
learns a per-layer vector at roughly one scale (0.45 to 1.04). Measured on the shipped
vectors, BiPO/CAA norm ratios are:

| group / model | layer 0 | layer 15 | layer 31 |
| --- | --- | --- | --- |
| persona-1, Llama-3.1-8B | 48-66x | 2.6-7.8x | 0.74-1.9x |
| persona-1, Mistral-7B | 96-180x | 8.0-13x | 4.3-7.3x |
| persona-2, Llama-3.1-8B | 240-737x | 5.0-11x | 1.9-5.4x |
| persona-2, Mistral-7B | 523-2211x | 14-36x | 1.1-2.4x |
| persona-3, Llama-3.1-8B | 5.9-13x | 0.16-0.40x | 0.042-0.11x |
| persona-3, Mistral-7B | 17-39x | 0.64-1.7x | 0.55-1.0x |

(The persona-2 rows are retained because those vector sets are committed, though that
group's data is not shipped and it cannot be run from this checkout.)

Since the persona-1 configs steer on the same multiplier sweep the CAA runs use, `raw`
would push one to three orders of magnitude harder than the CAA run at the early layers,
exactly where the stream is smallest and the relative perturbation therefore largest.

persona-3 is the exception worth knowing about: its ratio **crosses 1 around layer 5** and
keeps falling, so on Llama `raw` oversteers by about 6 to 13x at layer 0 and *under*steers
by about 10 to 24x at layer 31. The mismatch there is not a single direction that one
smaller multiplier would fix, which is why `reference` is the knob rather than the sweep.
`reference` puts both families on one axis; the applied factors are printed as a range per
behaviour at startup.

Two things it deliberately does not do. It never touches the direction (`neg` stays
exactly `-pos`, so both steering signs keep their meaning), and it matches the
*per-behaviour* norms rather than the norm of the averaged direction. That average's own
norm still falls where the group's directions put it: BiPO's members are less mutually
aligned than CAA's, so more cancels in the mean, and that difference is part of what the
run is measuring.

## Notes

- Training loads the policy **and** a frozen reference copy of the same checkpoint, so
  peak memory is roughly two models in bf16; only the steering vectors get gradients.
- The vectors are stored in bf16 to match `billy/vector/`, so a BiPO and a CAA vector set
  can be stacked in the same run; the optimiser's copy is fp32 while training.
- `bipo/trl/` shadows any installed `trl`: `bipo/` goes on `sys.path` ahead of it.
- There is no BiPO baseline config. The unsteered model is the same model, so
  `billy/configs/<persona>/<model>/baseline.yaml --baseline` covers both trees.
