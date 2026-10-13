# SYNC: A Training-Free Inference Controller for Dynamic Layer Selection in Multi-Persona Alignment

## Abstract

Steering vectors align a model at inference time without touching its weights, but
eliciting several personas at once is a different problem from eliciting one. Prior
multi-persona steering approaches predetermine both how persona vectors are combined and
the specific layers at which they are injected. Moreover, this static commitment
introduces substantial offline search costs to identify suitable target layers, despite
the fact that optimal intervention strengths vary dynamically across both decoding steps
and distinct personas. We introduce **SYNC** (**S**ensitivit**Y**-guided i**N**ference
**C**ontrol), a novel training-free inference control framework compatible with
pre-existing steering vectors. SYNC dynamically estimates each layer's sensitivity to
every persona via cosine similarity trajectories, assigning each persona to its optimal
injection layer at every decoding step. Across two models, two persona profiles and
steering vectors from both unsupervised and supervised extraction, SYNC leads on 21 of 28
personas, significantly on 15. We further show that a persona's steerability depends on
which personas are steered beside it, and that the cost of that falls unevenly across
them.

## How to run

### Unsupervised: `billy/`

```bash
# 1. extract the steering vectors, one behaviour per run
python billy/extract_caa.py   --config billy/configs/extract/llama3_1/power.yaml
python billy/extract_caa.py   --config billy/configs/extract/llama3_1/wealth.yaml
python billy/extract_caa.py   --config billy/configs/extract/llama3_1/coordinate.yaml
python billy/extract_caa.py   --config billy/configs/extract/llama3_1/corrigible.yaml

# 2. rank the layers  ->  billy/ablation/persona-1/llama-3/layer.csv
python ablation_layer.py      --config billy/configs/layer-ablation/llama3_1/persona-1.yaml

# 3. generate answers across the multiplier sweep
python generate_answers.py    --config billy/configs/persona-1/llama3_1/baseline.yaml --baseline
python generate_answers.py    --config billy/configs/persona-1/llama3_1/14-billy.yaml
python generate_answers.py    --config billy/configs/persona-1/llama3_1/simultaneous.yaml
python generate_answers.py    --config billy/configs/persona-1/llama3_1/sync-static.yaml
python generate_answers.py    --config billy/configs/persona-1/llama3_1/sync.yaml

# 4. score them with the LLM judge
python evaluation_llm.py      --config billy/configs/persona-1/llama3_1/baseline.yaml --baseline
python evaluation_llm.py      --config billy/configs/persona-1/llama3_1/14-billy.yaml
python evaluation_llm.py      --config billy/configs/persona-1/llama3_1/simultaneous.yaml
python evaluation_llm.py      --config billy/configs/persona-1/llama3_1/sync-static.yaml
python evaluation_llm.py      --config billy/configs/persona-1/llama3_1/sync.yaml
```

### Supervised: `bipo/`

```bash
# 1. train the steering vectors, one behaviour per run
python bipo/train.py          --config bipo/configs/train/llama3_1/power.yaml
python bipo/train.py          --config bipo/configs/train/llama3_1/wealth.yaml
python bipo/train.py          --config bipo/configs/train/llama3_1/coordinate.yaml
python bipo/train.py          --config bipo/configs/train/llama3_1/corrigible.yaml

# 2. rank the layers  ->  bipo/ablation/persona-1/llama-3/layer.csv
python ablation_layer.py      --config bipo/configs/layer-ablation/llama3_1/persona-1.yaml

# 3. generate answers across the multiplier sweep
python generate_answers.py    --config bipo/configs/persona-1/llama3_1/15-billy-ref.yaml
python generate_answers.py    --config bipo/configs/persona-1/llama3_1/simultaneous.yaml
python generate_answers.py    --config bipo/configs/persona-1/llama3_1/sync-static.yaml
python generate_answers.py    --config bipo/configs/persona-1/llama3_1/sync.yaml

# 4. score them with the LLM judge
python evaluation_llm.py      --config bipo/configs/persona-1/llama3_1/15-billy-ref.yaml
python evaluation_llm.py      --config bipo/configs/persona-1/llama3_1/simultaneous.yaml
python evaluation_llm.py      --config bipo/configs/persona-1/llama3_1/sync-static.yaml
python evaluation_llm.py      --config bipo/configs/persona-1/llama3_1/sync.yaml
```

