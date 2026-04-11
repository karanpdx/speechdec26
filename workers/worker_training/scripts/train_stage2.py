"""
CLI entry point for Stage 2 training.

Usage:
    python scripts/train_stage2.py --config configs/train_stage2.yaml
    python scripts/train_stage2.py --config configs/train_stage2.yaml --stage1-checkpoint checkpoints/stage1/best.pt

IMPORTANT: Run this ONLY after Stage 1 has converged and been validated.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2: decoder training with frozen encoders")
    parser.add_argument("--config", type=str, default="configs/train_stage2.yaml",
                        help="Path to train_stage2.yaml config file")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Override stage1_checkpoint from config")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from config")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    config_path = args.config

    if not Path(config_path).exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    final_config_path = config_path
    if args.stage1_checkpoint is not None or args.device is not None:
        with open(config_path, "r") as handle:
            config = yaml.safe_load(handle)
        if args.stage1_checkpoint is not None:
            config["stage1_checkpoint"] = args.stage1_checkpoint
        if args.device is not None:
            config["device"] = args.device

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config, handle)
            final_config_path = handle.name

    from src.training.train_stage2 import train
    train(config_path=final_config_path)


if __name__ == "__main__":
    main()
