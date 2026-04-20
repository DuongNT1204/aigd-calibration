# AIGD Calibration

Research code for the thesis topic:

> Calibration methods for AI-generated image detection under in-domain and cross-generator OOD settings.

The goal is to compare AI-generated image detectors not only by classification performance, but also by the reliability of their predicted probabilities.

This project supports two kinds of calibration experiments:

- **Train-time calibration:** Label Smoothing, Focal Loss, BSCE, Adaptive BSCE, Diff-DML.
- **Post-hoc calibration:** Temperature Scaling fitted on validation logits only.

## Project Status

This is a base research scaffold. It defines the project structure, PyTorch Lightning modules, CLI commands, config format, metrics, and artifact layout. It is not yet a finished benchmark with prepared datasets or trained checkpoints.

Package layout:

The source code imports modules as `aigd_calibration.*`. The Python package layout is:

```text
src/
└── aigd_calibration/
    ├── cli/
    ├── config/
    ├── data/
    ├── models/
    ├── methods/
    ├── lightning/
    ├── evaluation/
    ├── artifacts/
    ├── training/
    └── utils/
```

## Research Workflow

For the detailed experiment protocol, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The intended experiment flow is:

```text
metadata
  -> cross-generator splits
  -> train baseline detectors
  -> evaluate ID/OOD logits
  -> compute Accuracy/AUC/AP/F1/ECE/NLL/Brier
  -> apply calibration methods
  -> compare trade-off between discrimination and calibration
```

Typical study plan:

1. Train 3-4 base AIGD detectors, for example ResNet/CNN, ViT, CLIP, ConvNeXt.
2. Evaluate each baseline on `test_id` and `test_ood`.
3. Identify which models have good AUC but poor calibration.
4. Apply train-time calibration methods such as Label Smoothing, BSCE, Adaptive BSCE, and Diff-DML.
5. Apply post-hoc Temperature Scaling using validation logits only.
6. Compare ECE, Brier Score, NLL, AUC, AP, F1, and reliability diagrams.

## Repository Layout

```text
AIGD_Calibration/
├── configs/
│   ├── baselines/              # Training configs for baseline/calibrated models
│   ├── calibration/            # Post-hoc calibration configs
│   └── splits/                 # Dataset split configs
│
├── data/
│   ├── raw/                    # Optional local image storage
│   ├── metadata/               # metadata.jsonl
│   └── splits/                 # train/val/test_id/test_ood JSONL files
│
├── outputs/                    # Experiment outputs
│
└── src/aigd_calibration/
    ├── cli/                    # aigd split/train/eval/calibrate/report
    ├── config/                 # YAML loading and validation
    ├── data/                   # Dataset, DataModule, metadata, transforms, splits
    ├── models/                 # timm and CLIP model builders
    ├── methods/
    │   ├── train_time/         # Losses and Lightning strategies
    │   └── post_hoc/           # Temperature scaling, identity calibrator
    ├── lightning/              # Trainer, callbacks, loggers, checkpoints
    ├── evaluation/             # Metrics, reliability diagrams, reports
    ├── artifacts/              # Logits, tables, run directories
    ├── training/               # Optimizer/scheduler helpers
    └── utils/                  # IO, seed, device, logging helpers
```

## Data Format

The code is metadata-driven. Training does not depend on folder names directly.

Each image should have one metadata row:

```json
{"image_path": "/abs/path/to/image.png", "label": "synthetic", "generator": "stable_diffusion", "source": "my_dataset"}
```

Required fields:

- `image_path`: absolute path, or path relative to the split file.
- `label`: accepted real labels are `real` or `0`; accepted fake labels are `synthetic`, `fake`, `ai`, `generated`, or `1`.
- `generator`: generator/source name. This is required for clean cross-generator OOD splits.

Recommended data layout:

```text
data/
├── raw/
│   ├── real/
│   └── fake/
│       ├── stable_diffusion/
│       ├── midjourney/
│       └── dalle3/
├── metadata/
│   └── metadata.jsonl
└── splits/
    ├── train.jsonl
    ├── val.jsonl
    ├── test_id.jsonl
    └── test_ood.jsonl
```

The images may also live outside this repository. Only `image_path` must be correct.

## Installation

Create an environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If `pyproject.toml` is present, install the package in editable mode:

```bash
pip install -e .
```

Without editable install, run commands with:

```bash
PYTHONPATH=src python -m aigd_calibration.cli.main --help
```

After editable install, use:

```bash
aigd --help
```

## CLI Commands

For the full command reference, workflow, inputs, outputs, and leakage-safe
protocol, see:

```text
docs/CLI.md
```

The CLI has five main commands:

```bash
aigd split      # create train/val/test_id/test_ood JSONL splits
aigd train      # train a Lightning detector
aigd eval       # export logits and compute metrics
aigd calibrate  # fit/apply post-hoc calibration
aigd report     # collect metrics into a summary table
```

## Step 1: Create Cross-Generator Splits

Edit:

```text
configs/splits/cross_generator.yaml
```

Example:

```yaml
split:
  metadata: data/metadata/metadata.jsonl
  out_dir: data/splits
  ood_generators:
    - stable_diffusion
    - dalle3
  seed: 42
  val_ratio: 0.1
  test_id_ratio: 0.1
  ood_real_ratio: 0.1
```

Run:

```bash
aigd split --config configs/splits/cross_generator.yaml
```

This writes:

```text
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test_id.jsonl
data/splits/test_ood.jsonl
data/splits/summary.json
```

The OOD fake generators are held out from training.

## Step 2: Train Baselines

Example configs:

```text
configs/baselines/resnet50_bce.yaml
configs/baselines/vit_label_smoothing.yaml
configs/baselines/clip_bsce.yaml
configs/baselines/clip_diff_dml.yaml
```

Train one model:

```bash
aigd train --config configs/baselines/resnet50_bce.yaml
```

Train several models:

```bash
aigd train --config configs/baselines/resnet50_bce.yaml
aigd train --config configs/baselines/vit_label_smoothing.yaml
aigd train --config configs/baselines/clip_bsce.yaml
aigd train --config configs/baselines/clip_diff_dml.yaml
```

## Step 3: Evaluate ID and OOD

Export logits and compute metrics:

```bash
aigd eval --run outputs/resnet50_bce --split val
aigd eval --run outputs/resnet50_bce --split test_id
aigd eval --run outputs/resnet50_bce --split test_ood
```

Each evaluation writes:

```text
outputs/<experiment>/logits/<split>.jsonl
outputs/<experiment>/metrics/<split>.json
outputs/<experiment>/figures/reliability_<split>.png
```

Saved logits are the contract between evaluation, post-hoc calibration, and reporting. This prevents test-set leakage.

## Step 4: Post-Hoc Temperature Scaling

Temperature Scaling must be fitted on validation logits only.

Edit:

```text
configs/calibration/temperature.yaml
```

Example:

```yaml
calibration:
  method: temperature
  val_logits: outputs/resnet50_bce/logits/val.jsonl
  apply_logits:
    - outputs/resnet50_bce/logits/test_id.jsonl
    - outputs/resnet50_bce/logits/test_ood.jsonl
  out_dir: outputs/resnet50_bce/temperature
  init_temperature: 1.0
  max_iter: 100

metrics:
  ece_bins: 15
```

Run:

```bash
aigd calibrate --config configs/calibration/temperature.yaml
```

This writes calibrated logits and metrics under:

```text
outputs/resnet50_bce/temperature/
```

## Step 5: Build Summary Tables

Collect metrics across runs:

```bash
aigd report \
  --runs outputs/resnet50_bce outputs/clip_bsce outputs/clip_diff_dml \
  --output outputs/summary.csv
```

Use this CSV as the starting point for thesis tables.

## Metrics

Discrimination metrics:

- Accuracy
- AUC
- AP
- F1

Calibration metrics:

- ECE
- NLL
- Brier Score
- Reliability diagram

For the thesis, compare both groups together. A model can have high AUC but poor ECE/Brier under OOD.

## Method Categories

Train-time methods:

```text
standard + BCE/CE
standard + Label Smoothing
standard + Focal Loss
standard + BSCE
standard + Adaptive BSCE
Diff-DML two-model training
```

Post-hoc methods:

```text
Identity baseline
Temperature Scaling
```

Diff-DML is implemented as a training strategy, not as a loss, because it needs two models and KL agreement during optimization. To follow the paper, keep the primary model `f` on a normal decaying LR schedule and keep the auxiliary model `g` on a lower fixed LR:

```yaml
optimizer:
  lr: 0.0001

scheduler:
  name: cosine

optimizer_aux:
  lr: 0.00001

scheduler_aux:
  name: none
```

## Output Layout

Each experiment should produce:

```text
outputs/<experiment>/
├── config.yaml
├── checkpoints/
│   ├── last.ckpt
│   └── *.ckpt
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
└── tables/
```

## Common Mistakes to Avoid

- Do not fit Temperature Scaling on `test_id` or `test_ood`.
- Do not allow OOD fake generators to appear in `train.jsonl`.
- Do not compare methods using different data splits.
- Do not report only AUC; calibration metrics are the main research focus.
- Do not change model, data, loss, augmentation, and split all at once unless the experiment is explicitly designed that way.

## Current Next Steps

1. Install dependencies.
2. Create `data/metadata/metadata.jsonl`.
3. Generate splits with `aigd split`.
4. Start with `resnet50_bce`, then add CLIP/ViT baselines.
5. Evaluate ID/OOD before applying calibration methods.
