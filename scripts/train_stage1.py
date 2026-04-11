"""
CLI entry point for Stage 1 training.

Usage:
    python scripts/train_stage1.py --config configs/train_stage1.yaml
    python scripts/train_stage1.py --config configs/train_stage1.yaml --device cpu
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1: joint multi-modal contrastive learning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_stage1.yaml",
        help="Path to train_stage1.yaml config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (e.g., 'cpu', 'cuda', 'cuda:1')",
    )
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
        logging.error("Config file not found: %s", config_path)
        sys.exit(1)

    final_config_path = config_path
    if args.device is not None:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["device"] = args.device

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config, handle)
            final_config_path = handle.name

    from src.training.train_stage1 import train

    train(config_path=final_config_path)


if __name__ == "__main__":
    main()
