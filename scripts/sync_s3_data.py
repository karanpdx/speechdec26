"""
Stage raw modality data from S3 into local directories.

Usage:
    python scripts/sync_s3_data.py --config configs/s3_data.yaml --dry-run
    python scripts/sync_s3_data.py --config configs/s3_data.yaml --modalities eeg fmri
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Sync raw modality data from S3 into local staging directories")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/s3_data.yaml",
        help="Path to S3 staging config",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        choices=["eeg", "fmri", "meg"],
        help="Subset of modalities to sync",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sync commands without executing them",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_source_uri(config: dict, source: str) -> str:
    if source.startswith("s3://"):
        return source.rstrip("/")

    bucket = config.get("bucket")
    prefix = str(config.get("prefix") or "").strip("/")
    if not bucket:
        raise ValueError(
            "S3 config must define either a full s3:// source URI for each dataset "
            "or a top-level `bucket` value."
        )

    parts = [f"s3://{bucket}"]
    if prefix:
        parts.append(prefix)
    parts.append(source.strip("/"))
    return "/".join(parts)


def build_sync_plan(config: dict, selected_modalities: list[str] | None = None) -> list[dict]:
    datasets = config.get("datasets", {})
    selected = selected_modalities or sorted(datasets.keys())
    plan = []
    for modality in selected:
        if modality not in datasets:
            raise ValueError(f"Unknown modality '{modality}' in S3 sync plan")
        dataset_cfg = datasets[modality]
        local_path = dataset_cfg.get("local_path")
        source = dataset_cfg.get("source")
        if not local_path or not source:
            raise ValueError(f"Dataset '{modality}' must define both source and local_path")
        plan.append(
            {
                "modality": modality,
                "source_uri": _resolve_source_uri(config, str(source)),
                "local_path": str(local_path),
            }
        )
    return plan


def _aws_base_command(config: dict) -> list[str]:
    cmd = ["aws"]
    profile = config.get("aws_profile")
    region = config.get("aws_region")
    if profile:
        cmd.extend(["--profile", str(profile)])
    if region:
        cmd.extend(["--region", str(region)])
    return cmd


def build_sync_command(config: dict, source_uri: str, local_path: str) -> list[str]:
    return _aws_base_command(config) + ["s3", "sync", source_uri, local_path]


def run_sync_plan(config: dict, plan: list[dict], dry_run: bool = False) -> None:
    if shutil.which("aws") is None:
        raise RuntimeError("AWS CLI is not installed or not on PATH")

    for item in plan:
        target = Path(item["local_path"])
        target.mkdir(parents=True, exist_ok=True)
        cmd = build_sync_command(config, item["source_uri"], item["local_path"])
        logging.info("%s → %s", item["source_uri"], item["local_path"])
        logging.info("Command: %s", " ".join(cmd))
        if dry_run:
            continue
        subprocess.run(cmd, check=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_config(str(config_path))
    plan = build_sync_plan(config, args.modalities)
    run_sync_plan(config, plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
