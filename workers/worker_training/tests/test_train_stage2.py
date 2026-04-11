"""
Tests for Stage 2 training helpers and smoke wiring.
"""

import csv
import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_common_config(paths: dict, checkpoint_path: str, checkpoint_dir: str) -> dict:
    return {
        "stage1_checkpoint": checkpoint_path,
        "split_file": paths["split_path"],
        "vocab_embeddings": paths["vocab_path"],
        "processed_base_dir": paths["processed_dir"],
        "modalities": ["eeg", "meg", "fmri"],
        "embed_dim": 768,
        "eeg_channels": 8,
        "eeg_timepoints": 32,
        "meg_channels": 10,
        "meg_timepoints": 32,
        "fmri_voxels": 12,
        "subject_embed_dim": 16,
        "n_subjects": 4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "n_epochs": 2,
        "grad_clip": 1.0,
        "device": "cpu",
        "mse_weight": 1.0,
        "freq_domain_weight": 0.1,
        "spatial_smooth_weight": 0.05,
        "spatial_smooth_max_dist_mm": 6.0,
        "checkpoint_dir": checkpoint_dir,
        "log_every_n_steps": 1,
        "val_every_n_epochs": 1,
    }


def _write_stage1_checkpoint(base_dir: str, config: dict) -> str:
    from src.training.train_stage1 import build_models, build_optimizer, save_checkpoint

    models = build_models(config)
    optimizer = build_optimizer(models, config)
    checkpoint_dir = str(Path(base_dir) / "checkpoints" / "stage1")
    return save_checkpoint(
        models=models,
        optimizer=optimizer,
        epoch=0,
        metrics={"top1_eeg": 0.0},
        checkpoint_dir=checkpoint_dir,
        is_best=True,
    )


def test_load_stage1_checkpoint_freezes_encoders_and_keeps_subject_embedding_trainable():
    from stubs.data_stubs import write_stub_dataset
    from src.training.train_stage1 import build_models
    from src.training.train_stage2 import load_stage1_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        paths = write_stub_dataset(
            base_dir=tmp,
            n_subjects=4,
            n_epochs_per_subject=4,
            n_channels_eeg=8,
            n_channels_meg=10,
            n_voxels=12,
            n_timepoints=32,
            n_words=6,
        )
        base_config = _make_common_config(paths, checkpoint_path="", checkpoint_dir=str(Path(tmp) / "unused"))
        checkpoint_path = _write_stage1_checkpoint(tmp, base_config)
        base_config["stage1_checkpoint"] = checkpoint_path

        models = build_models(base_config)
        encoders = load_stage1_checkpoint(checkpoint_path, models, "cpu")

        for encoder in encoders.values():
            assert all(not param.requires_grad for param in encoder.parameters())
        assert all(not param.requires_grad for param in models["projector"].parameters())
        assert any(param.requires_grad for param in models["subject_emb"].parameters())


def test_spatial_smoothness_loss_empty_edges_returns_zero_with_gradient():
    from src.training.train_stage2 import spatial_smoothness_loss

    pred = torch.randn(3, 5, requires_grad=True)
    adjacency = torch.empty((0, 2), dtype=torch.long)

    loss = spatial_smoothness_loss(pred, adjacency)
    assert loss.item() == 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad == 0)


def test_build_voxel_adjacency_finds_only_close_pairs():
    from src.training.train_stage2 import build_voxel_adjacency

    coords = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 4],
            [0, 0, 10],
        ],
        dtype=torch.int32,
    )

    adjacency = build_voxel_adjacency(coords, max_dist_mm=6.0)
    pairs = {tuple(pair.tolist()) for pair in adjacency}
    assert pairs == {(0, 1), (1, 2)}


def test_load_stage1_checkpoint_missing_file_raises():
    from src.training.train_stage2 import load_stage1_checkpoint

    with pytest.raises(FileNotFoundError):
        load_stage1_checkpoint("/tmp/does-not-exist-stage1.pt", {}, "cpu")


def test_stage2_train_smoke_writes_csv_and_checkpoint():
    from stubs.data_stubs import write_stub_dataset
    from src.training.train_stage2 import train

    with tempfile.TemporaryDirectory() as tmp:
        paths = write_stub_dataset(
            base_dir=tmp,
            n_subjects=4,
            n_epochs_per_subject=4,
            n_channels_eeg=8,
            n_channels_meg=10,
            n_voxels=12,
            n_timepoints=32,
            n_words=6,
            modalities=["eeg", "meg", "fmri"],
        )
        checkpoint_dir = str(Path(tmp) / "checkpoints" / "stage2")
        base_config = _make_common_config(paths, checkpoint_path="", checkpoint_dir=checkpoint_dir)
        checkpoint_path = _write_stage1_checkpoint(tmp, base_config)
        stage2_config = _make_common_config(paths, checkpoint_path, checkpoint_dir)

        config_path = Path(tmp) / "train_stage2_test.yaml"
        with open(config_path, "w") as handle:
            yaml.safe_dump(stage2_config, handle)

        train(str(config_path))

        log_path = Path(checkpoint_dir) / "train_log.csv"
        assert log_path.exists()
        with open(log_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "Stage 2 train_log.csv should contain train/val rows"
        for row in rows:
            for key in (
                "total_loss",
                "mse_loss",
                "freq_loss",
                "smooth_loss",
                "eeg_loss",
                "meg_loss",
                "fmri_loss",
            ):
                value = float(row[key])
                assert not math.isnan(value)
                assert not math.isinf(value)

        checkpoint_files = list(Path(checkpoint_dir).glob("*.pt"))
        assert checkpoint_files, "Stage 2 training should write at least one checkpoint"
