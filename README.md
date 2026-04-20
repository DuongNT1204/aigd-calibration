# AIGD Calibration

Research code for the thesis:

> **Calibration methods for AI-generated image detection under in-domain and cross-generator OOD settings.**

The goal is to compare AIGD detectors not only by classification performance (AUC, Accuracy, MCC) but also by the reliability of their predicted probabilities (ECE, Brier Score, NLL).

## Hypothesis

> A CE-trained baseline produces overconfident, unreliable probabilities on unseen generators (high ECE, high Brier). Applying train-time calibration methods reduces ECE and Brier while maintaining or improving accuracy.

**Validated** — see [`outputs/results.md`](outputs/results.md) for full results.

---

## Repository Layout

```text
AIGD_Calibration/
├── configs/
│   ├── baselines/              # Training configs for each experiment
│   │   ├── clip_ce.yaml        # CE loss (baseline)
│   │   ├── clip_bce.yaml       # Binary CE
│   │   ├── clip_focal.yaml     # Focal Loss (γ=2.0)
│   │   ├── clip_bsce.yaml      # Balanced Softmax CE
│   │   ├── clip_bsce_adaptive.yaml
│   │   ├── clip_label_smoothing.yaml
│   │   └── clip_diff_dml.yaml  # Differentiated Deep Mutual Learning
│   ├── calibration/
│   │   └── temperature.yaml    # Post-hoc temperature scaling
│   └── splits/
│       └── cross_generator.yaml
├── data/
│   ├── raw/                    # Image files (not tracked)
│   ├── metadata/               # metadata.jsonl
│   └── splits/                 # train/val/test_id/test_ood JSONL
├── docs/
│   ├── CLI.md                  # Full CLI reference
│   └── EXPERIMENTS.md          # Experiment protocol and results
├── outputs/                    # Experiment outputs (not tracked)
│   └── results.md              # Summary of results and analysis
└── src/aigd_calibration/
    ├── cli/                    # aigd split/train/eval/calibrate/report
    ├── config/                 # YAML loading
    ├── data/                   # Dataset, DataModule, transforms, splits
    ├── models/                 # CLIP and timm model builders
    ├── methods/
    │   ├── train_time/         # Losses + strategies (standard, diff_dml)
    │   └── post_hoc/           # Temperature scaling
    ├── lightning/              # Trainer, callbacks, loggers, checkpoints
    ├── evaluation/             # Metrics, reliability diagrams, reports
    ├── artifacts/              # Logits, tables, run directories
    ├── training/               # Optimizer/scheduler builders
    └── utils/                  # IO, seed, device, logging
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA 12.4 (tested with 2x RTX 3090)

### Step 1: Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Or with conda:

```bash
conda create -n aigd python=3.10 -y
conda activate aigd
```

### Step 2: Install PyTorch with CUDA 12.4

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> If you have a different CUDA version, replace `cu124` with the appropriate tag (e.g. `cu121`, `cu118`).
> Check your driver version with `nvidia-smi`.

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
pip install tensorboard
```

### Step 4: Install the package in editable mode

```bash
pip install -e .
```

Verify:

```bash
aigd --help
```

---

## Data Format

The code is metadata-driven. Each image needs one JSONL row:

```json
{"image_path": "/abs/path/to/image.png", "label": 1, "generator": "stable_diffusion_v1_4", "source": "GenImage"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `image_path` | yes | Absolute or relative path to the image |
| `label` | yes | `0` or `real` for real; `1`, `synthetic`, or `fake` for AI-generated |
| `generator` | yes | Generator name — used for OOD splits |
| `source` | optional | Dataset name |

### Recommended data layout

```text
data/
├── raw/
│   ├── real/
│   └── fake/
│       ├── stable_diffusion_v1_4/
│       ├── midjourney/
│       └── biggan/
├── metadata/
│   └── metadata.jsonl
└── splits/
    ├── train.jsonl
    ├── val.jsonl
    ├── test_id.jsonl
    └── test_ood.jsonl
```

---

## Workflow

### 1. Create splits

```bash
aigd split --config configs/splits/cross_generator.yaml
```

Edit `configs/splits/cross_generator.yaml` to set which generators are held out for OOD:

```yaml
split:
  metadata: data/metadata/metadata.jsonl
  out_dir: data/splits
  ood_generators:
    - Midjourney
    - BigGAN
    - GLIDE
  seed: 42
  val_ratio: 0.1
  test_id_ratio: 0.1
  ood_real_ratio: 0.1
```

### 2. Train

```bash
aigd train --config configs/baselines/clip_ce.yaml
```

### 3. Evaluate

```bash
aigd eval --run outputs/clip_ce --split val
aigd eval --run outputs/clip_ce --split test_id
aigd eval --run outputs/clip_ce --split test_ood
```

### 4. Post-hoc calibration (Temperature Scaling)

Edit `configs/calibration/temperature.yaml` to point to your run, then:

```bash
aigd calibrate --config configs/calibration/temperature.yaml
```

### 5. Report

```bash
aigd report \
  --runs outputs/clip_ce outputs/clip_focal outputs/clip_label_smoothing \
  --output outputs/summary.csv
```

For the full CLI reference, see [`docs/CLI.md`](docs/CLI.md).
For the experiment protocol, see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## Experiment Setup

All experiments use **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`, 303M params) with partial fine-tuning:

- Freeze first 16/24 transformer blocks, train the remaining 8 blocks + linear head
- AdamW: `lr_head=1e-4`, `lr_backbone=1e-5`, `weight_decay=1e-4`
- 2 epochs, batch size 32, 32-bit precision, 2x RTX 3090 (DDP)

| Config | Loss / Strategy |
|--------|----------------|
| `clip_ce` | Cross-Entropy (baseline) |
| `clip_bce` | Binary Cross-Entropy |
| `clip_focal` | Focal Loss (gamma=2.0) |
| `clip_bsce` | Balanced Softmax CE |
| `clip_bsce_adaptive` | Adaptive BSCE |
| `clip_label_smoothing` | Label Smoothing (epsilon=0.1) |
| `clip_diff_dml` | Diff-DML: two-model KL mutual learning |

---

## Key Results (Test-OOD)

Full analysis and per-split tables in [`outputs/results.md`](outputs/results.md).

| Method | ACC | ECE | Brier | NLL |
|--------|-----|-----|-------|-----|
| CE (baseline) | 0.750 | 0.153 | 0.199 | 0.770 |
| BCE | 0.753 | 0.146 | 0.195 | 0.763 |
| Focal | 0.767 | **0.023** | 0.161 | 0.488 |
| BSCE | 0.787 | 0.048 | 0.155 | 0.488 |
| BSCE Adaptive | 0.796 | 0.048 | 0.150 | 0.473 |
| Label Smoothing | 0.823 | 0.032 | **0.123** | **0.388** |
| Diff-DML | **0.829** | 0.050 | 0.131 | 0.425 |

All train-time calibration methods reduce ECE by 3-7x and Brier by 20-38% vs CE baseline, while improving OOD accuracy by 2-8%.

---

## Output Layout

```text
outputs/<experiment_name>/
├── config.yaml
├── checkpoints/
│   ├── last.ckpt
│   └── epoch=...ckpt
├── logits/
│   ├── val.jsonl
│   ├── test_id.jsonl
│   └── test_ood.jsonl
├── metrics/
│   ├── val.json
│   ├── test_id.json
│   └── test_ood.json
├── figures/
│   ├── reliability_val.png
│   ├── reliability_test_id.png
│   └── reliability_test_ood.png
├── logs/               # CSVLogger
├── tb_logs/            # TensorBoard
└── temperature/
    ├── calibrator.json
    ├── test_id_calibrated.jsonl
    ├── test_id_metrics.json
    ├── test_ood_calibrated.jsonl
    └── test_ood_metrics.json
```

Monitor training:

```bash
tensorboard --logdir outputs/<experiment_name>/tb_logs
```

---

## Metrics

| Metric | Type | Direction |
|--------|------|-----------|
| Accuracy | Classification | higher is better |
| MCC | Classification | higher is better |
| AUC | Classification | higher is better |
| AP | Classification | higher is better |
| F1 | Classification | higher is better |
| ECE | Calibration | lower is better |
| Brier | Calibration | lower is better |
| NLL | Calibration | lower is better |

---

## Rules

- Never fit temperature scaling on `test_id` or `test_ood` — only on `val`.
- Never allow OOD generators to appear in `train.jsonl`.
- Use the same splits for all method comparisons.
- Report both classification and calibration metrics.
