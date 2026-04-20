# Dataset Guide

This project uses metadata files to describe the dataset. The training code does not rely on folder names directly; it reads JSONL rows that point to image files and define labels/generators.

## Recommended Dataset

For the first thesis experiments, use **GenImage** or a smaller subset such as Tiny-GenImage.

GenImage already includes both:

- `ai/`: AI-generated images
- `nature/`: real images from ImageNet

So you do **not** need to collect real images separately for the initial experiments.

Typical GenImage layout:

```text
GenImage/
├── Stable Diffusion V1.4/
│   ├── train/
│   │   ├── ai/
│   │   └── nature/
│   └── val/
│       ├── ai/
│       └── nature/
├── Midjourney/
│   ├── train/
│   │   ├── ai/
│   │   └── nature/
│   └── val/
│       ├── ai/
│       └── nature/
└── BigGAN/
    ├── train/
    │   ├── ai/
    │   └── nature/
    └── val/
        ├── ai/
        └── nature/
```

GenImage generators usually include:

```text
BigGAN
ADM
GLIDE
VQDM
Stable Diffusion V1.4
Stable Diffusion V1.5
Wukong
Midjourney
```

## Local Data Layout

You can store images inside this repo:

```text
data/
├── raw/
│   └── GenImage/
├── metadata/
│   └── metadata.jsonl
└── splits/
    ├── train.jsonl
    ├── val.jsonl
    ├── test_id.jsonl
    └── test_ood.jsonl
```

Or you can store images elsewhere:

```text
/mnt/datasets/GenImage/
/workspace/datasets/GenImage/
```

Both are fine. The only requirement is that `image_path` in the JSONL files points to the correct image.

## Metadata Format

Each row in `data/metadata/metadata.jsonl` should describe one image:

```json
{"image_path": "/path/to/image.png", "label": "synthetic", "generator": "stable_diffusion_v14", "source": "genimage"}
```

Real image example:

```json
{"image_path": "/path/to/Stable Diffusion V1.4/train/nature/image_001.jpg", "label": "real", "generator": "real", "source": "genimage"}
```

Fake image example:

```json
{"image_path": "/path/to/Stable Diffusion V1.4/train/ai/image_001.png", "label": "synthetic", "generator": "stable_diffusion_v14", "source": "genimage"}
```

Accepted labels:

```text
real labels: real, 0
fake labels: synthetic, fake, ai, generated, 1
```

Use consistent generator names. Recommended names:

```text
biggan
adm
glide
vqdm
stable_diffusion_v14
stable_diffusion_v15
wukong
midjourney
```

## Recommended First Split

For the first real experiment, train on one generator and test OOD on other generators.

Recommended:

```text
train:
  Stable Diffusion V1.4/train/nature  -> real
  Stable Diffusion V1.4/train/ai      -> fake

val:
  Stable Diffusion V1.4/val/nature    -> real
  Stable Diffusion V1.4/val/ai        -> fake

test_id:
  held-out part of Stable Diffusion V1.4/val/nature
  held-out part of Stable Diffusion V1.4/val/ai

test_ood:
  Midjourney/val/nature + Midjourney/val/ai
  BigGAN/val/nature + BigGAN/val/ai
  GLIDE/val/nature + GLIDE/val/ai
  ADM/val/nature + ADM/val/ai
  VQDM/val/nature + VQDM/val/ai
  Wukong/val/nature + Wukong/val/ai
  Stable Diffusion V1.5/val/nature + Stable Diffusion V1.5/val/ai
```

This answers the core research question:

```text
If a detector is trained on one generator, how reliable are its probabilities on unseen generators?
```

## Debug Dataset

Before training full models, create a small debug split:

```text
train:
  200 real + 200 SD1.4 fake

val:
  100 real + 100 SD1.4 fake

test_id:
  100 real + 100 SD1.4 fake

test_ood:
  100 real + 100 Midjourney fake
  100 real + 100 BigGAN fake
  100 real + 100 GLIDE fake
```

Use this to check:

- dataset loading
- augmentation pipeline
- Lightning training loop
- checkpoint saving
- logit export
- ECE / NLL / Brier computation
- Temperature Scaling

## Thesis-Scale Dataset

After the debug split works, use a larger subset:

```text
train:
  5k-20k real + 5k-20k SD1.4 fake

val:
  1k real + 1k SD1.4 fake

test_id:
  1k-5k real + 1k-5k SD1.4 fake

test_ood:
  1k-5k real/fake pairs per unseen generator
```

Keep the number of real and fake samples balanced for each split when possible.

## Important Notes

- `train` and `test_ood` must not share fake generators.
- Do not fit Temperature Scaling on `test_id` or `test_ood`.
- Use `val` only for checkpoint selection and post-hoc calibration fitting.
- Keep split files fixed for all methods so comparisons are fair.
- Prefer absolute paths in metadata if your dataset is outside this repository.
- Be careful with format bias: if all real images are JPEG and all fake images are PNG, models may learn file/compression artifacts instead of generation artifacts.

## Output Split Files

The training code expects:

```text
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test_id.jsonl
data/splits/test_ood.jsonl
```

Each split file uses the same JSONL row format as `metadata.jsonl`.
