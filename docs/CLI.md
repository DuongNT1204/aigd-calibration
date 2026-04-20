# CLI Guide

This project exposes one command-line entry point:

```bash
aigd
```

The CLI is designed around the full experiment lifecycle:

```text
metadata -> split -> train -> eval -> calibrate -> report
```

Use this guide when you want to run experiments without touching Python code.

## Installation

From the repository root (use your project's venv):

```bash
pip install -e .
```

After editable install, check that the CLI is available:

```bash
aigd --help
```

If you do not install the package, run commands with:

```bash
PYTHONPATH=src python -m aigd_calibration.cli.main --help
```

## Commands

The CLI has five commands:

```bash
aigd split      # create train/val/test_id/test_ood JSONL splits
aigd train      # train one experiment from a YAML config
aigd eval       # export logits, metrics, and reliability diagram
aigd calibrate  # fit post-hoc calibration on val logits and apply to test logits
aigd report     # collect metrics from runs into a CSV table
```

Global option:

```bash
aigd --log-level INFO <command> ...
```

Common log levels are `DEBUG`, `INFO`, `WARNING`, and `ERROR`.

## Expected Data Contract

The training and evaluation code reads split files in JSONL format. Each row
should describe one image:

```json
{"image_path": "data/raw/GenImage/stable_diffusion_v_1_4/nature/000001.png", "label": 0, "generator": "real", "source": "stable_diffusion_v_1_4"}
{"image_path": "data/raw/GenImage/stable_diffusion_v_1_4/ai/000001.png", "label": 1, "generator": "stable_diffusion_v_1_4", "source": "GenImage"}
```

Required fields:

```text
image_path  path to the image file
label       0 for real, 1 for fake
```

Recommended fields:

```text
generator   generator name, e.g. stable_diffusion_v_1_4, Midjourney, BigGAN
source      dataset/source name, e.g. GenImage
```

The dataset loader also accepts aliases such as `path` or `file` for
`image_path`, and `model` or `dataset` for `generator`.

## Config Files

The most complete reference config is:

```text
configs/baselines/base_config.yaml
```

Use it as the source of truth for available config keys. For a real experiment,
copy it and edit the copy:

```bash
cp configs/baselines/base_config.yaml configs/baselines/my_resnet50_bce.yaml
```

Existing example configs:

```text
configs/splits/cross_generator.yaml
configs/baselines/resnet50_bce.yaml
configs/baselines/resnet50_bce_ddp.yaml
configs/baselines/vit_label_smoothing.yaml
configs/baselines/clip_bsce.yaml
configs/baselines/clip_diff_dml.yaml
configs/calibration/temperature.yaml
```

## Full Workflow

### 1. Create Splits

Use:

```bash
aigd split --config configs/splits/cross_generator.yaml
```

The split config controls:

```yaml
split:
  metadata: data/metadata/all.jsonl
  out_dir: data/splits
  ood_generators:
    - Midjourney
    - BigGAN
  seed: 42
  val_ratio: 0.1
  test_id_ratio: 0.1
  ood_real_ratio: 0.1
```

Output:

```text
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test_id.jsonl
data/splits/test_ood.jsonl
data/splits/summary.json
```

Meaning:

```text
train     in-domain real/fake images used for optimization
val       in-domain validation images used for checkpointing and calibration fitting
test_id   in-domain test images used for final ID evaluation
test_ood  held-out generator test images used for OOD evaluation
```

You can also run split creation without a config:

```bash
aigd split \
  --metadata data/metadata/all.jsonl \
  --out-dir data/splits \
  --ood-generators Midjourney,BigGAN \
  --seed 42 \
  --val-ratio 0.1 \
  --test-id-ratio 0.1 \
  --ood-real-ratio 0.1
```

### 2. Train A Model

Use:

```bash
aigd train --config configs/baselines/resnet50_bce.yaml
```

The training command reads:

```text
experiment
data
model
model_aux          only for Diff-DML
training
optimizer
scheduler
optimizer_aux     only for Diff-DML
scheduler_aux     only for Diff-DML
loss
metrics
```

Output:

```text
outputs/<experiment.name>/
├── config.yaml
├── checkpoints/
│   ├── last.ckpt
│   └── epoch=...ckpt
├── logs/               # CSVLogger metrics
└── tb_logs/            # TensorBoard logs (run: tensorboard --logdir tb_logs)
```

Example standard baseline:

```yaml
experiment:
  name: resnet50_bce

model:
  type: timm
  name: resnet50
  pretrained: true
  num_classes: 2

training:
  strategy: standard
  epochs: 10

loss:
  name: bce
```

Example Diff-DML:

```yaml
experiment:
  name: clip_diff_dml

training:
  strategy: diff_dml
  kl_weight: 1.0

model:
  type: clip
  name: openai/clip-vit-large-patch14
  freeze_backbone: true
  num_frozen_blocks: 16

model_aux:
  type: clip
  name: openai/clip-vit-large-patch14
  freeze_backbone: true
  num_frozen_blocks: 20

optimizer:
  lr: 0.0001

scheduler:
  name: cosine

optimizer_aux:
  lr: 0.00001

scheduler_aux:
  name: none
```

For Diff-DML, `model` is the primary model `f`, and `model_aux` is the
auxiliary model `g`. Only `model_f` is used at inference.

### 3. Evaluate A Trained Run

Use:

```bash
aigd eval --run outputs/resnet50_bce --split val
aigd eval --run outputs/resnet50_bce --split test_id
aigd eval --run outputs/resnet50_bce --split test_ood
```

Optional checkpoint:

```bash
aigd eval \
  --run outputs/resnet50_bce \
  --split test_id \
  --checkpoint outputs/resnet50_bce/checkpoints/last.ckpt
```

If `--checkpoint` is omitted, the code finds a checkpoint from the run folder.

Output per split:

```text
outputs/<experiment.name>/
├── logits/
│   ├── val.jsonl
│   ├── test_id.jsonl
│   └── test_ood.jsonl
├── metrics/
│   ├── val.json
│   ├── test_id.json
│   └── test_ood.json
└── figures/
    ├── reliability_val.png
    ├── reliability_test_id.png
    └── reliability_test_ood.png
```

The logits files are important. They are the bridge between raw evaluation and
post-hoc calibration.

### 4. Fit Temperature On Val And Apply To Test

First, make sure validation and test logits exist:

```bash
aigd eval --run outputs/resnet50_bce --split val
aigd eval --run outputs/resnet50_bce --split test_id
aigd eval --run outputs/resnet50_bce --split test_ood
```

Then edit:

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
  threshold: 0.5
```

Run:

```bash
aigd calibrate --config configs/calibration/temperature.yaml
```

What happens:

```text
1. Load validation logits from calibration.val_logits
2. Fit one temperature value on validation NLL
3. Save calibrator.json
4. Apply the learned temperature to test_id/test_ood logits
5. Save calibrated logits and calibrated metrics
```

Output:

```text
outputs/resnet50_bce/temperature/
├── calibrator.json
├── test_id_calibrated.jsonl
├── test_id_metrics.json
├── test_ood_calibrated.jsonl
└── test_ood_metrics.json
```

Do not fit temperature on `test_id` or `test_ood`. Those splits are only for
evaluation. Fitting on test data would leak test information into the method.

> **Note:** Temperature scaling only works meaningfully on a converged model. If
> the model is undertrained (e.g. only 1 epoch), the optimizer may push temperature
> to its minimum bound (0.05), making calibration worse instead of better. Always
> run temperature scaling on fully trained checkpoints.

### 5. Build A Summary Table

Use:

```bash
aigd report \
  --runs outputs/resnet50_bce outputs/clip_bsce outputs/clip_diff_dml \
  --output outputs/summary.csv
```

Output:

```text
outputs/summary.csv
```

Columns in order: `run, split, calibration, accuracy, mcc, f1, ap, auc, nll, brier, ece, count`

- `calibration` is `none` for raw model metrics, `temperature` for temperature-scaled metrics.
- The report collects both raw (`metrics/*.json`) and calibrated (`temperature/*_metrics.json`) rows automatically.

This CSV is the starting point for thesis tables.

## Command Reference

### `aigd split`

Purpose:

```text
Create train/val/test_id/test_ood JSONL files from a metadata file.
```

Required if using config:

```bash
aigd split --config configs/splits/cross_generator.yaml
```

Required if not using config:

```bash
aigd split --metadata data/metadata/all.jsonl --out-dir data/splits
```

Options:

```text
--config           YAML config containing a split section
--metadata         input metadata JSONL/JSON/CSV
--out-dir          output directory for split JSONL files
--ood-generators   comma-separated fake generators held out for OOD
--seed             split seed
--val-ratio        validation fraction from in-domain data
--test-id-ratio    test_id fraction from in-domain data
--ood-real-ratio   real-image fraction added to test_ood
```

### `aigd train`

Purpose:

```text
Train one configured experiment.
```

Usage:

```bash
aigd train --config configs/baselines/resnet50_bce.yaml
```

Required:

```text
--config   experiment YAML config
```

Important config sections:

```text
experiment.name   output run name
data              split paths and loader settings
model             primary model
model_aux         auxiliary model for Diff-DML
training          Lightning and strategy settings
optimizer         optimizer for standard/model_f
scheduler         scheduler for standard/model_f
optimizer_aux     optimizer for Diff-DML model_g
scheduler_aux     scheduler for Diff-DML model_g
loss              train-time calibration loss
metrics           ECE bins and classification threshold
```

### `aigd eval`

Purpose:

```text
Run prediction on one split, save logits, compute metrics, and plot reliability.
```

Usage:

```bash
aigd eval --run outputs/resnet50_bce --split test_id
```

Options:

```text
--run          run directory containing config.yaml and checkpoints
--split        val, test_id, or test_ood
--checkpoint   optional checkpoint path
```

Notes:

```text
val logits are needed for post-hoc calibration fitting.
test_id and test_ood logits are needed for final reporting.
```

### `aigd calibrate`

Purpose:

```text
Fit post-hoc calibration on validation logits, then apply it to test logits.
```

Usage:

```bash
aigd calibrate --config configs/calibration/temperature.yaml
```

Required:

```text
--config   calibration YAML config
```

Important config fields:

```text
calibration.method         identity, temperature, or temperature_scaling
calibration.val_logits     validation logits used to fit calibrator
calibration.apply_logits   one or more logits files to calibrate
calibration.out_dir        output directory
calibration.max_iter       LBFGS iterations for temperature scaling
metrics.ece_bins           ECE bin count for calibrated metrics
metrics.threshold          binary decision threshold for calibrated metrics
```

### `aigd report`

Purpose:

```text
Collect metrics from multiple run directories into one CSV.
```

Usage:

```bash
aigd report \
  --runs outputs/resnet50_bce outputs/clip_bsce \
  --output outputs/summary.csv
```

Options:

```text
--runs     one or more run directories
--output   output CSV path
```

## Common Experiment Recipes

### Standard Baseline

```bash
aigd train --config configs/baselines/resnet50_bce.yaml
aigd eval --run outputs/resnet50_bce --split val
aigd eval --run outputs/resnet50_bce --split test_id
aigd eval --run outputs/resnet50_bce --split test_ood
```

### Baseline + Temperature Scaling

```bash
aigd eval --run outputs/resnet50_bce --split val
aigd eval --run outputs/resnet50_bce --split test_id
aigd eval --run outputs/resnet50_bce --split test_ood
aigd calibrate --config configs/calibration/temperature.yaml
```

### Diff-DML

```bash
aigd train --config configs/baselines/clip_diff_dml.yaml
aigd eval --run outputs/clip_diff_dml --split val
aigd eval --run outputs/clip_diff_dml --split test_id
aigd eval --run outputs/clip_diff_dml --split test_ood
```

### Multi-GPU DDP

Use a config with:

```yaml
training:
  accelerator: gpu
  devices: 2
  lightning_strategy: ddp
  sync_batchnorm: false
  use_distributed_sampler: true
```

Then run normally:

```bash
aigd train --config configs/baselines/resnet50_bce_ddp.yaml
```

## Output Layout

A complete run usually looks like:

```text
outputs/<experiment.name>/
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
└── temperature/
    ├── calibrator.json
    ├── test_id_calibrated.jsonl
    ├── test_id_metrics.json
    ├── test_ood_calibrated.jsonl
    └── test_ood_metrics.json
```

## Metrics

Raw and calibrated evaluation report these columns:

```text
accuracy   classification accuracy at threshold (default 0.5)
mcc        Matthews Correlation Coefficient — more robust than accuracy under class imbalance
f1         F1 score at threshold
ap         Average Precision (area under precision-recall curve)
auc        ROC-AUC — threshold-invariant discrimination metric
nll        Negative Log-Likelihood — penalises confident wrong predictions heavily
brier      Brier Score = mean squared error of fake probability vs label
ece        Expected Calibration Error — gap between confidence and accuracy
count      number of samples
```

Calibration metrics (lower is better):

```text
ece    brier    nll
```

Discrimination metrics (higher is better):

```text
accuracy    mcc    f1    ap    auc
```

## Rules To Avoid Leakage

Use this protocol for clean experiments:

```text
train     optimize model weights
val       select checkpoint and fit post-hoc temperature
test_id   final in-domain evaluation only
test_ood  final held-out generator evaluation only
```

Never fit temperature, choose checkpoints, or tune hyperparameters on
`test_id` or `test_ood`.

## Troubleshooting

If `aigd` is not found:

```bash
pip install -e .
```

or use:

```bash
PYTHONPATH=src python -m aigd_calibration.cli.main --help
```

If training cannot find images, check that every split row has a valid
`image_path`.

If `aigd calibrate` fails, check that you already ran `aigd eval` for `val` and
for every split listed in `calibration.apply_logits`.

If DDP hangs or behaves unexpectedly, first test with:

```yaml
training:
  accelerator: gpu
  devices: 1
  lightning_strategy: auto
```

Then switch back to DDP after the single-GPU run works.

If `aigd eval` produces duplicate rows in logits (count > expected), it may be
a leftover from DDP predict. The eval command forces single-GPU inference
automatically — this should not occur after the current version.

If temperature scaling worsens NLL (i.e. temperature hits the minimum bound
0.05), the model is likely undertrained. Run more epochs before calibrating.

To view training metrics interactively:

```bash
tensorboard --logdir outputs/<experiment.name>/tb_logs
```
