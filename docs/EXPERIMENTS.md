# Experiment Protocol

This document describes the experiment plan for calibration-aware AI-generated image detection.

The main research question is:

```text
When an AIGD detector is tested on unseen generators, does its confidence remain reliable?
```

The experiments should always compare two groups of metrics:

- **Classification / discrimination:** can the model separate real and fake images?
- **Calibration / reliability:** are the predicted probabilities trustworthy?

## Controlled Experiment Design

For fair comparison, all models are trained on the same data splits
(train/val/test_id/test_ood) with the same augmentation pipeline and
evaluation protocol. Optimizer settings (learning rate, scheduler) are
tuned per backbone to ensure each model is trained under its best-known
configuration. The only variable changed between experiments within the
same backbone is the loss function or training strategy.

This is a deliberate design choice: the thesis focuses on the effect of
**calibration methods**, not augmentation differences. Fixing the data
splits and augmentation pipeline isolates calibration as the variable of
interest, which is the standard approach for ablation studies in ML papers.

## Core Evaluation Settings

Every method should be evaluated on the same splits:

```text
train       used to train the detector
val         used for checkpoint selection and post-hoc calibration fitting
test_id     in-domain test, same fake generator family as train
test_ood    cross-generator OOD test, fake generators unseen during training
```

Example with GenImage:

```text
train:
  Stable Diffusion V1.4/train/nature
  Stable Diffusion V1.4/train/ai

val:
  Stable Diffusion V1.4/val/nature
  Stable Diffusion V1.4/val/ai

test_id:
  held-out Stable Diffusion V1.4 real/fake samples

test_ood:
  Midjourney, BigGAN, GLIDE, ADM, VQDM, Wukong, Stable Diffusion V1.5
```

The important rule:

```text
Fake generators in test_ood must not appear in train.
```

## Experiment Groups

Run experiments in this order.

## 1. Basic Training Without Calibration Method

Purpose:

```text
Establish baseline detector performance before applying calibration methods.
```

Train 3-4 base detectors with normal CE/BCE:

```text
ResNet50 + BCE/CE
ViT + BCE/CE
CLIP + BCE/CE
optional ConvNeXt/EVA/DINOv2 + BCE/CE
```

Evaluate each baseline on:

```text
val
test_id
test_ood
```

Questions to answer:

- Which baseline has the best AUC/AP/F1?
- Which baseline has the lowest ECE/Brier/NLL?
- Does `test_ood` calibration become worse than `test_id` calibration?
- Is there a case where AUC is still good but ECE/Brier is poor?

Example commands:

```bash
aigd train --config configs/baselines/resnet50_bce.yaml

aigd eval --run outputs/resnet50_bce --split val
aigd eval --run outputs/resnet50_bce --split test_id
aigd eval --run outputs/resnet50_bce --split test_ood
```

Minimum table:

```text
Model | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
```

## 2. Post-Hoc Calibration

Purpose:

```text
Check whether a simple calibration layer can improve reliability without retraining the detector.
```

Post-hoc methods operate on saved logits.

Start with:

```text
Temperature Scaling
```

Protocol:

```text
1. Train baseline model.
2. Export val logits.
3. Fit temperature on val logits only.
4. Apply the learned temperature to test_id and test_ood logits.
5. Recompute metrics.
```

Do not fit temperature on `test_id` or `test_ood`.

Example commands:

```bash
aigd eval --run outputs/resnet50_bce --split val
aigd eval --run outputs/resnet50_bce --split test_id
aigd eval --run outputs/resnet50_bce --split test_ood

aigd calibrate --config configs/calibration/temperature.yaml
```

Questions to answer:

- Does Temperature Scaling reduce ECE?
- Does it reduce NLL?
- Does it improve Brier Score?
- Does it preserve AUC/AP?
- Is temperature fitted on in-domain validation still effective under OOD?

Expected behavior:

```text
AUC and AP usually stay almost unchanged because temperature scaling preserves logit ranking.
ECE, NLL, and Brier may improve if miscalibration is mostly overconfidence/underconfidence.
```

Minimum comparison:

```text
Model | Calibration | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
ResNet50 | none | test_ood | ...
ResNet50 | temperature | test_ood | ...
```

## 3. Train-Time Calibration Methods

Purpose:

```text
Check whether changing the training objective or training strategy improves calibration directly.
```

Train-time methods require retraining the model.

Recommended methods:

```text
Label Smoothing
Focal Loss
BSCE
Adaptive BSCE
Diff-DML
```

### 3.1 Label Smoothing

Type:

```text
Loss regularization
```

Reason:

```text
Prevents the model from becoming too confident in one-hot labels.
```

Compare against:

```text
same model + CE/BCE
```

### 3.2 Focal Loss

Type:

```text
Loss reweighting
```

Reason:

```text
Changes sample weighting based on difficulty. It is useful as a calibration-related baseline, although its original purpose is class imbalance.
```

### 3.3 BSCE

Type:

```text
Brier-score-weighted CE
```

Reason:

```text
Weights training using probability error / uncertainty information.
```

### 3.4 Adaptive BSCE

Type:

```text
Adaptive Brier-aware loss
```

Reason:

```text
Uses different weighting behavior depending on model confidence.
```

### 3.5 Diff-DML

Type:

```text
Training strategy
```

Reason:

```text
Trains a primary model and an auxiliary model together using KL agreement.
Only the primary model is used at inference.
```

Diff-DML is not just a loss. It changes the training loop because it needs:

```text
model_f
model_g
loss_f
loss_g
KL divergence
manual optimization
```

In the Diff-DML paper, `model_f` and `model_g` are intentionally trained with
different optimization behavior. The primary model `f` is the supervised model
used at inference and follows the normal decaying LR schedule. The auxiliary
model `g` is used only during training, removes CE supervision, and keeps a
fixed lower LR. This is the Differentiated Training Strategy (DTS).

Questions to answer:

- Which train-time method gives the best ECE?
- Which gives the best Brier Score?
- Which preserves classification performance best?
- Does the method help more on `test_id` or `test_ood`?
- Does calibration improvement come with lower AUC/F1?

Minimum table:

```text
Model | Train Method | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
CLIP  | BCE           | test_ood | ...
CLIP  | LabelSmooth   | test_ood | ...
CLIP  | BSCE          | test_ood | ...
CLIP  | Diff-DML      | test_ood | ...
```

## 4. Train-Time Method + Post-Hoc Calibration

Purpose:

```text
Check whether post-hoc calibration still helps after train-time calibration.
```

Example combinations:

```text
CLIP + Label Smoothing + Temperature Scaling
CLIP + BSCE + Temperature Scaling
CLIP + Diff-DML + Temperature Scaling
ViT + Label Smoothing + Temperature Scaling
```

Protocol:

```text
1. Train model with train-time method.
2. Export val/test_id/test_ood logits.
3. Fit temperature on val logits.
4. Apply temperature to test_id and test_ood.
5. Compare before and after temperature.
```

Questions to answer:

- If train-time calibration already improves ECE, does Temperature Scaling improve it further?
- Does Temperature Scaling help all methods equally?
- Is the best final method a train-time method alone, or train-time + post-hoc?

Minimum table:

```text
Model | Train Method | Post-hoc | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
CLIP  | BSCE         | none     | test_ood | ...
CLIP  | BSCE         | temp     | test_ood | ...
```

## Metrics

Use the same metrics for every experiment.

## Classification Metrics

### Accuracy

Measures the fraction of correct predictions.

Good for simple interpretation, but can be misleading if classes are imbalanced.

### MCC

Matthews Correlation Coefficient.

Recommended because it is more informative than accuracy under imbalance.

Range:

```text
-1 bad
 0 random
 1 perfect
```

### AUC

Area under ROC curve.

Measures ranking quality across thresholds.

Important note:

```text
AUC does not measure calibration.
```

### AP

Average Precision.

Useful when fake/real balance changes.

### F1

Harmonic mean of precision and recall at one threshold.

Usually threshold is:

```text
prob_fake >= 0.5
```

## Calibration Metrics

### ECE

Expected Calibration Error.

Measures the gap between confidence and empirical accuracy across confidence bins.

Lower is better.

### NLL

Negative Log-Likelihood.

Penalizes wrong confident predictions strongly.

Lower is better.

### Brier Score

Mean squared error between predicted fake probability and true label.

For binary AIGD:

```text
Brier = mean((prob_fake - label)^2)
```

Lower is better.

### Reliability Diagram

Visual plot of confidence vs accuracy.

Use it to show:

```text
overconfidence: confidence > accuracy
underconfidence: confidence < accuracy
```

## Recommended Final Result Tables

### Table 1: Baseline Comparison

```text
Model | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
```

### Table 2: Post-Hoc Calibration

```text
Model | Post-hoc | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
```

### Table 3: Train-Time Calibration

```text
Model | Train Method | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
```

### Table 4: Train-Time + Post-Hoc

```text
Model | Train Method | Post-hoc | Split | Acc | MCC | AUC | AP | F1 | ECE | NLL | Brier
```

## Recommended Figures

Use at least:

```text
Reliability diagram for baseline on test_id
Reliability diagram for baseline on test_ood
Reliability diagram before/after Temperature Scaling
Bar chart: ECE by method on test_ood
Scatter plot: AUC vs ECE
Scatter plot: AUC vs Brier
```

## Fairness Rules

To make the comparison valid:

- Use the same `train`, `val`, `test_id`, and `test_ood` splits for all methods.
- Use the same model backbone when comparing calibration methods.
- Use the same checkpoint selection metric, preferably validation AUC.
- Fit post-hoc calibration only on validation logits.
- Report both classification and calibration metrics.
- Keep random seeds fixed where possible.
- Do not tune methods on `test_ood`.

## Multi-GPU Training

The project uses PyTorch Lightning, so multi-GPU training is handled by the Lightning Trainer.

Use DDP by setting:

```yaml
training:
  accelerator: gpu
  devices: 2
  lightning_strategy: ddp
  use_distributed_sampler: true
```

Notes:

- `data.batch_size` is the per-GPU batch size under DDP.
- Effective global batch size is `batch_size * devices`.
- If global batch size changes a lot, consider scaling the learning rate.
- Start with `sync_batchnorm: false`; enable it only if you specifically need synchronized BatchNorm.
- See `configs/baselines/resnet50_bce_ddp.yaml` for a ready-to-copy example.

## Backbone Freezing

For `timm` models:

```yaml
model:
  freeze_backbone: true
```

freezes the whole backbone and trains only the classification head.

For CLIP models, partial transformer-block freezing is supported:

```yaml
model:
  type: clip
  freeze_backbone: true
  num_frozen_blocks: 16
  train_layer_norm: true
```

Meaning:

```text
num_frozen_blocks: null -> freeze all CLIP vision layers, train only head
num_frozen_blocks: 0    -> train all CLIP transformer blocks
num_frozen_blocks: 16   -> freeze first 16 blocks, train remaining blocks
num_frozen_blocks: 24   -> freeze all 24 blocks for CLIP ViT-L/14
```

For Diff-DML, primary and auxiliary models can use different values, for example:

```yaml
model:
  primary:
    num_frozen_blocks: 16
  auxiliary:
    num_frozen_blocks: 20
```

Diff-DML should use separate optimizer and scheduler configs for the primary
model `f` and auxiliary model `g` when following the paper:

```yaml
optimizer:
  lr_backbone: 0.00001
  lr_head: 0.0001

optimizer_aux:
  lr_backbone: 0.000001
  lr_head: 0.00001

scheduler:
  name: cosine

scheduler_aux:
  name: none
```

For a closer reproduction of the paper's CIFAR schedule, use `multistep` for
`model_f`:

```yaml
scheduler:
  name: multistep
  milestones: [100, 150, 200]
  gamma: 0.1

scheduler_aux:
  name: none
```

If `optimizer_aux` or `scheduler_aux` is omitted, the auxiliary model reuses
`optimizer` or `scheduler`, which is convenient but less faithful to Diff-DML's
DTS idea.

## Optimizer And Scheduler

The optimizer builder supports separate parameter groups for backbone and head,
similar to the BitMind training code:

```yaml
optimizer:
  name: adamw
  lr: 0.0001
  lr_backbone: 0.00001
  lr_head: 0.0001
  weight_decay: 0.0001
  weight_decay_head: 0.00001
```

Behavior:

```text
parameters with "head" in the module path -> lr_head, weight_decay_head
other trainable parameters               -> lr_backbone, weight_decay
frozen parameters                         -> ignored by optimizer
```

This is useful for pretrained backbones:

```text
small LR for CLIP/ResNet/ViT backbone
larger LR for the new classification head
```

Schedulers:

```yaml
scheduler:
  name: none
```

keeps LR constant.

```yaml
scheduler:
  name: warmup_cosine
  warmup_ratio: 0.03
  eta_min: 0.000001
```

uses linear warmup followed by cosine restarts over training steps.

## Actual Experiment Results

All 7 experiments below were run with CLIP ViT-L/14, 2 epochs, 2x RTX 3090.
Full tables and analysis: `outputs/results.md`.

### Test-OOD (cross-generator, the most important split)

| Method | ACC | ECE | Brier | NLL |
|--------|-----|-----|-------|-----|
| CE (baseline) | 0.750 | 0.153 | 0.199 | 0.770 |
| BCE | 0.753 | 0.146 | 0.195 | 0.763 |
| Focal | 0.767 | **0.023** | 0.161 | 0.488 |
| BSCE | 0.787 | 0.048 | 0.155 | 0.488 |
| BSCE Adaptive | 0.796 | 0.048 | 0.150 | 0.473 |
| Label Smoothing | 0.823 | 0.032 | **0.123** | **0.388** |
| Diff-DML | **0.829** | 0.050 | 0.131 | 0.425 |

Key findings:
- CE baseline has the worst calibration (ECE=0.153) and worst OOD accuracy (0.750)
- All train-time methods reduce ECE by 3-7x and Brier by 20-38%
- Label Smoothing best overall: highest ACC on both ID and OOD, lowest Brier
- Focal Loss has the best ECE on OOD but moderate accuracy
- Diff-DML achieves the highest OOD ACC but lower ID accuracy — mutual learning regularizes for distribution shift
- Temperature scaling helps on test_id but often hurts on test_ood (temperature fit on in-domain val does not generalize)

### Suggested Experiment Order

For a new thesis run:

```text
1. CLIP + CE         (baseline — establishes ECE/Brier baseline)
2. CLIP + Focal      (best ECE on OOD)
3. CLIP + Label Smoothing  (best overall trade-off)
4. CLIP + BSCE / BSCE Adaptive
5. CLIP + Diff-DML   (best OOD accuracy)
6. Temperature Scaling on all runs
7. Report with aigd report
```
