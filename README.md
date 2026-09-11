# Rule-Grounded Construction SGG

This repository provides the reference implementation of the rule-grounded scene graph learning framework described in **“Rule-Grounded Scene Graph Learning for Structurally Consistent Construction Safety Relation Understanding.”**

It contains the model, rule configuration, joint objective, post-hoc validation, evaluation protocol, and a synthetic smoke-test example. Reported numerical results are provided in the paper.

## Scope

- dense ordered pairs for every detected object pair with distinct endpoints;
- 66-D pair input: 16-D subject ROI appearance, 16-D object ROI appearance, 16-D union-region ROI appearance, and 18-D pair geometry;
- 128-D MP-GNN relation reasoning with three message-passing blocks;
- optional three-block Transformer relation backbone with learned subject/object category embeddings;
- 128-D rule encoding from categories, the 5-D legality mask, and 8-D spatial-semantic evidence;
- forward concatenation of 128-D graph and rule tokens before five-class relation classification;
- PPE assignment using a Softmax over candidate workers and cross-entropy;
- zone, interaction, and legality consistency losses;
- overall F1, R@K, mR@K, SCR, and RVR;
- inference-only post-hoc validation for the `+ Validate` variants.

The joint Rule-Grounded objective is:

`L = Lrel + 0.2 Lppe + 0.1 Lzone + 0.1 Lint + 0.1 Llegal`.

## Layout

```text
configs/                     paper-default MP-GNN and Transformer configs
docs/                        data format and evaluation protocol
examples/                    synthetic graph used for smoke testing
scripts/train.py             Baseline or Rule-Grounded training entry
scripts/evaluate.py          aggregate evaluation entry
src/rule_grounded_sgg/       model, rules, assignment, losses, and metrics
tests/                       compact synthetic consistency tests
```

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e .
```

The paper reports Python 3.12.8, PyTorch 2.7.0, CUDA 12.8, and an NVIDIA RTX 4080 GPU. The code also supports CPU execution for small tests.

## Smoke test

```bash
python scripts/train.py \
  --config configs/paper_default.yaml \
  --train examples/synthetic_graphs.json \
  --val examples/synthetic_graphs.json \
  --output-dir outputs/smoke \
  --mode rule_grounded \
  --epochs 1

python scripts/evaluate.py \
  --config configs/paper_default.yaml \
  --checkpoint outputs/smoke/best.pt \
  --data examples/synthetic_graphs.json
```

Use `--mode baseline` for the visual baseline, `configs/transformer.yaml` for the Transformer comparison, and `--posthoc` with `evaluate.py` for the `+ Validate` variant.

## Data availability

The implementation code and evaluation protocol are provided in this repository. The construction-site images and annotations used in the study cannot be publicly shared because of data-use and privacy restrictions.

See [docs/data_format.md](docs/data_format.md) and [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) for the input format and evaluation details.
