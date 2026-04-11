"""
Tests for validate and save_checkpoint in src/training/train_stage1.py.
TDD red-phase tests — written before implementation.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "train_stage1.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["device"] = "cpu"
    return cfg


@pytest.fixture
def models(config):
    from src.training.train_stage1 import build_models
    return build_models(config)


@pytest.fixture
def optimizer(models, config):
    from src.training.train_stage1 import build_optimizer
    return build_optimizer(models, config)


@pytest.fixture
def vocab_setup():
    V, D = 20, 768
    vocab_embs = np.random.randn(V, D).astype(np.float32)
    vocab = [f"word_{i:03d}" for i in range(V)]
    return vocab_embs, vocab


def make_batch(B=4, V=20, D=768):
    return {
        "eeg": {
            "data": torch.randn(B, 64, 175),
            "bert_emb": torch.randn(B, D),
            "subject_idx": torch.zeros(B, dtype=torch.long),
            "label_idx": torch.randint(0, V, (B,)),
        },
        "shared_label_mask": torch.ones(B, dtype=torch.bool),
    }


# ---------- validate tests ----------

class TestValidate:
    def test_returns_dict_with_top1_eeg(self, models, vocab_setup):
        from src.training.train_stage1 import validate
        vocab_embs, vocab = vocab_setup
        batch = make_batch()
        metrics = validate(models, [batch], vocab_embs, vocab)
        assert "top1_eeg" in metrics

    def test_top1_value_in_0_1(self, models, vocab_setup):
        from src.training.train_stage1 import validate
        vocab_embs, vocab = vocab_setup
        batch = make_batch()
        metrics = validate(models, [batch], vocab_embs, vocab)
        assert 0.0 <= metrics["top1_eeg"] <= 1.0

    def test_empty_dataloader_returns_empty_dict(self, models, vocab_setup):
        from src.training.train_stage1 import validate
        vocab_embs, vocab = vocab_setup
        metrics = validate(models, [], vocab_embs, vocab)
        assert metrics == {}

    def test_encoders_restored_to_train_mode(self, models, vocab_setup):
        from src.training.train_stage1 import validate
        vocab_embs, vocab = vocab_setup
        batch = make_batch()
        validate(models, [batch], vocab_embs, vocab)
        assert models["eeg_encoder"].training
        assert models["projector"].training

    def test_no_grad_does_not_accumulate_gradients(self, models, vocab_setup):
        from src.training.train_stage1 import validate
        vocab_embs, vocab = vocab_setup
        batch = make_batch()
        validate(models, [batch], vocab_embs, vocab)
        # Parameters should have no grad after validate
        for p in models["eeg_encoder"].parameters():
            assert p.grad is None


# ---------- save_checkpoint tests ----------

class TestSaveCheckpoint:
    def test_creates_epoch_file(self, models, optimizer):
        from src.training.train_stage1 import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = str(Path(tmp) / "checkpoints" / "stage1")
            path = save_checkpoint(models, optimizer, epoch=0,
                                   metrics={"top1_eeg": 0.25},
                                   checkpoint_dir=ckpt_dir)
            assert Path(path).exists()
            assert "epoch_0" in path

    def test_no_best_when_not_best(self, models, optimizer):
        from src.training.train_stage1 import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = str(Path(tmp) / "checkpoints" / "stage1")
            save_checkpoint(models, optimizer, epoch=0,
                            metrics={}, checkpoint_dir=ckpt_dir, is_best=False)
            assert not (Path(ckpt_dir) / "best.pt").exists()

    def test_creates_best_pt_when_is_best(self, models, optimizer):
        from src.training.train_stage1 import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = str(Path(tmp) / "checkpoints" / "stage1")
            save_checkpoint(models, optimizer, epoch=1,
                            metrics={"top1_eeg": 0.5},
                            checkpoint_dir=ckpt_dir, is_best=True)
            assert (Path(ckpt_dir) / "best.pt").exists()

    def test_checkpoint_contains_required_keys(self, models, optimizer):
        from src.training.train_stage1 import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = str(Path(tmp) / "stage1")
            path = save_checkpoint(models, optimizer, epoch=5,
                                   metrics={"top1_eeg": 0.1},
                                   checkpoint_dir=ckpt_dir)
            state = torch.load(path, map_location="cpu", weights_only=False)
            assert "model_state_dicts" in state
            assert "optimizer_state_dict" in state
            assert state["epoch"] == 5
            assert state["metrics"] == {"top1_eeg": 0.1}

    def test_creates_checkpoint_dir_if_missing(self, models, optimizer):
        from src.training.train_stage1 import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = str(Path(tmp) / "deep" / "nested" / "dir")
            path = save_checkpoint(models, optimizer, epoch=0,
                                   metrics={}, checkpoint_dir=ckpt_dir)
            assert Path(ckpt_dir).exists()
            assert Path(path).exists()

    def test_returns_string_path(self, models, optimizer):
        from src.training.train_stage1 import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(models, optimizer, epoch=3,
                                   metrics={}, checkpoint_dir=tmp)
            assert isinstance(path, str)
