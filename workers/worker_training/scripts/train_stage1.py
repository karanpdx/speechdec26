"""
CLI entry point for Stage 1 training.

Usage:
    python scripts/train_stage1.py --config configs/train_stage1.yaml
    python scripts/train_stage1.py --config configs/train_stage1.yaml --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1: joint multi-modal contrastive learning")
    parser.add_argument("--config", type=str, default="configs/train_stage1.yaml",
                        help="Path to train_stage1.yaml config file")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from config (e.g., 'cpu', 'cuda', 'cuda:1')")
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

    # Import here so errors surface cleanly
    from src.training.train_stage1 import train
    train(config_path=config_path)


if __name__ == "__main__":
    main()
