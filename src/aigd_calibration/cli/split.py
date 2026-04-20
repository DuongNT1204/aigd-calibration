"""CLI command for creating cross-generator splits."""

from __future__ import annotations

import argparse
import logging


def add_parser(subparsers) -> None:
    """Register `aigd split`."""
    parser = subparsers.add_parser("split", help="Create train/val/test_id/test_ood splits")
    parser.add_argument("--config", help="Optional YAML split config")
    parser.add_argument("--metadata", help="Input metadata JSONL/JSON/CSV")
    parser.add_argument("--out-dir", help="Output split directory")
    parser.add_argument("--ood-generators", default="", help="Comma-separated fake generators held out for OOD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-id-ratio", type=float, default=0.1)
    parser.add_argument("--ood-real-ratio", type=float, default=0.1)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run split creation."""
    from aigd_calibration.config.loader import load_config
    from aigd_calibration.data.splits import write_cross_generator_splits

    if args.config:
        cfg = load_config(args.config)
        split_cfg = cfg.get("split", cfg)
        metadata = split_cfg["metadata"]
        out_dir = split_cfg["out_dir"]
        ood_generators = set(split_cfg.get("ood_generators", []))
        seed = int(split_cfg.get("seed", args.seed))
        val_ratio = float(split_cfg.get("val_ratio", args.val_ratio))
        test_id_ratio = float(split_cfg.get("test_id_ratio", args.test_id_ratio))
        ood_real_ratio = float(split_cfg.get("ood_real_ratio", args.ood_real_ratio))
    else:
        metadata = args.metadata
        out_dir = args.out_dir
        ood_generators = {item.strip() for item in args.ood_generators.split(",") if item.strip()}
        seed = args.seed
        val_ratio = args.val_ratio
        test_id_ratio = args.test_id_ratio
        ood_real_ratio = args.ood_real_ratio

    if not metadata or not out_dir:
        raise ValueError("Provide --metadata and --out-dir, or use --config.")

    summary = write_cross_generator_splits(metadata, out_dir, ood_generators, seed, val_ratio, test_id_ratio, ood_real_ratio)
    logging.info("Created splits: %s", summary)
