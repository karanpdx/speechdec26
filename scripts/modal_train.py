"""
Modal runner for the root alignment and training entry points.

Examples:
    modal run scripts/modal_train.py --operation alignment --config configs/splits.yaml
    modal run scripts/modal_train.py --operation stage1 --config configs/train_stage1.yaml
    modal run scripts/modal_train.py --operation stage2 --config configs/train_stage2.yaml --stage1-checkpoint checkpoints/stage1/best.pt

Stage 1 and Stage 2 are configured to use an A10G GPU in the Modal app definition.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in import-safe tests
    modal = None


APP_NAME = "speechdec26-train"
WORKDIR = "/workspace"
DATA_VOLUME_NAME = "speechdec26-data"
CHECKPOINT_VOLUME_NAME = "speechdec26-checkpoints"
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24


def _write_runtime_config(
    config_path: str,
    *,
    device: str | None = None,
    stage1_checkpoint: str | None = None,
) -> str:
    """Write a temporary config with runtime overrides and return its path."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if device is not None:
        config["device"] = device
    if stage1_checkpoint is not None:
        config["stage1_checkpoint"] = stage1_checkpoint

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return handle.name


def _run_alignment(config_path: str) -> dict:
    from src.data.align_splits import run_alignment

    split_path, vocab_path = run_alignment(config_path)
    return {"split_path": split_path, "vocab_path": vocab_path}


def _run_stage1(config_path: str, *, device: str | None = None) -> dict:
    from src.training.train_stage1 import train

    runtime_config = _write_runtime_config(config_path, device=device)
    train(runtime_config)
    return {"config_path": runtime_config}


def _run_stage2(
    config_path: str,
    *,
    device: str | None = None,
    stage1_checkpoint: str | None = None,
) -> dict:
    if not stage1_checkpoint:
        raise ValueError("stage1_checkpoint is required for Stage 2")

    from src.training.train_stage2 import train

    runtime_config = _write_runtime_config(
        config_path,
        device=device,
        stage1_checkpoint=stage1_checkpoint,
    )
    train(runtime_config)
    return {
        "config_path": runtime_config,
        "stage1_checkpoint": stage1_checkpoint,
    }


def _require_modal():
    if modal is None:
        raise RuntimeError(
            "The `modal` package is not installed. Install it locally to use this runner, "
            "for example: `pip install modal`."
        )


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install_from_requirements("requirements.txt")
        .add_local_dir("src", remote_path=f"{WORKDIR}/src")
        .add_local_dir("scripts", remote_path=f"{WORKDIR}/scripts")
        .add_local_dir("workers", remote_path=f"{WORKDIR}/workers")
        .add_local_dir("configs", remote_path=f"{WORKDIR}/configs")
        .add_local_file("requirements.txt", remote_path=f"{WORKDIR}/requirements.txt")
    )

    app = modal.App(APP_NAME)
    DATA_VOLUME = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
    CHECKPOINT_VOLUME = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

    def _common_kwargs(gpu: str | None, timeout_seconds: int) -> dict:
        kwargs = {
            "image": image,
            "timeout": timeout_seconds,
            "volumes": {
                f"{WORKDIR}/data": DATA_VOLUME,
                f"{WORKDIR}/checkpoints": CHECKPOINT_VOLUME,
            },
        }
        if gpu and gpu.lower() != "none":
            kwargs["gpu"] = gpu
        return kwargs

    @app.function(**_common_kwargs(gpu=None, timeout_seconds=DEFAULT_TIMEOUT_SECONDS))
    def run_alignment_remote(config_path: str) -> dict:
        os.chdir(WORKDIR)
        result = _run_alignment(config_path)
        DATA_VOLUME.commit()
        CHECKPOINT_VOLUME.commit()
        return result

    @app.function(**_common_kwargs(gpu="A10G", timeout_seconds=DEFAULT_TIMEOUT_SECONDS))
    def run_stage1_remote(config_path: str, device: str = "cuda") -> dict:
        os.chdir(WORKDIR)
        result = _run_stage1(config_path, device=device)
        CHECKPOINT_VOLUME.commit()
        return result

    @app.function(**_common_kwargs(gpu="A10G", timeout_seconds=DEFAULT_TIMEOUT_SECONDS))
    def run_stage2_remote(
        config_path: str,
        stage1_checkpoint: str,
        device: str = "cuda",
    ) -> dict:
        os.chdir(WORKDIR)
        result = _run_stage2(
            config_path,
            device=device,
            stage1_checkpoint=stage1_checkpoint,
        )
        CHECKPOINT_VOLUME.commit()
        return result

    @app.local_entrypoint()
    def main(
        operation: str,
        config: str,
        stage1_checkpoint: str = "",
        device: str = "cuda",
    ):
        """
        Run alignment, Stage 1, or Stage 2 on Modal.

        Examples:
            modal run scripts/modal_train.py --operation stage1 --config configs/train_stage1.yaml
            modal run scripts/modal_train.py --operation stage2 --config configs/train_stage2.yaml --stage1-checkpoint checkpoints/stage1/best.pt

        Stage 1 and Stage 2 use the GPU configured on the Modal function definition.
        """
        if operation == "alignment":
            print(run_alignment_remote.remote(config))
            return
        if operation == "stage1":
            print(run_stage1_remote.remote(config, device=device))
            return
        if operation == "stage2":
            if not stage1_checkpoint:
                raise ValueError("--stage1-checkpoint is required for operation=stage2")
            print(run_stage2_remote.remote(config, stage1_checkpoint=stage1_checkpoint, device=device))
            return
        raise ValueError("operation must be one of: alignment, stage1, stage2")

else:
    app = None
