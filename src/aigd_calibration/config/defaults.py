"""Default values shared across experiment configs.

The project keeps defaults in one place so CLI commands can fill missing YAML
keys without each command inventing its own behavior.
"""

DEFAULT_CONFIG = {
    "seed": 42,
    "output_root": "outputs",
    "data": {
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 4,
        "normalization": "imagenet",
    },
    "training": {
        "strategy": "standard",
        "epochs": 10,
        "accelerator": "auto",
        "devices": "auto",
        "lightning_strategy": "auto",
        "sync_batchnorm": False,
        "use_distributed_sampler": True,
        "precision": "32-true",
        "log_every_n_steps": 20,
    },
    "optimizer": {
        "name": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
    },
    "loss": {
        "name": "ce",
    },
    "metrics": {
        "ece_bins": 15,
        "threshold": 0.5,
    },
}
