"""
CLI entry point for dataset alignment and vocabulary generation.

Usage:
    python scripts/run_alignment.py --config configs/splits.yaml
"""

import argparse
import logging
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate aligned train/val/test splits and vocabulary embeddings")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/splits.yaml",
        help="Path to splits.yaml config file",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    if not Path(args.config).exists():
        logging.error("Config file not found: %s", args.config)
        sys.exit(1)

    from src.data.align_splits import run_alignment

    run_alignment(args.config)


if __name__ == "__main__":
    main()
