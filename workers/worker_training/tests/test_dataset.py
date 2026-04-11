"""
Tests for MultiModalDataset — plans 02-01 and 02-02.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure package root on path when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stubs.data_stubs import write_stub_dataset
from src.training.dataset import MultiModalDataset


@pytest.fixture(scope="module")
def stub_paths():
    with tempfile.TemporaryDirectory() as d:
        paths = write_stub_dataset(
            d,
            n_subjects=4,
            n_epochs_per_subject=20,
            n_words=10,
            modalities=["eeg", "meg", "fmri"],
        )
        yield paths


# ---------------------------------------------------------------------------
# Task 1: __init__ and flat index building
# ---------------------------------------------------------------------------


def test_dataset_instantiates(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    assert len(ds) > 0, "Dataset must not be empty"


def test_dataset_filters_to_requested_modalities(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
        modalities=["eeg"],
    )
    modalities_seen = {ds[i]["modality"] for i in range(len(ds))}
    assert modalities_seen == {"eeg"}, f"Expected only eeg, got {modalities_seen}"


def test_missing_file_does_not_raise(tmp_path):
    """A split JSON that points to non-existent files should instantiate without raising."""
    import json

    split_data = {
        "train": {"eeg": ["missing_sub.npz"]},
        "val": {"eeg": []},
        "test": {"eeg": []},
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split_data))

    # Build a minimal vocab file
    from stubs.data_stubs import make_vocab_embeddings

    vocab_data = make_vocab_embeddings(n_words=5)
    vocab_path = tmp_path / "vocab_embeddings.npz"
    np.savez(str(vocab_path), **vocab_data)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Should not raise — missing file is skipped with a warning
    ds = MultiModalDataset(str(split_path), str(vocab_path), str(processed_dir), split="train")
    assert len(ds) == 0


def test_subject_ids_populated(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    subject_ids = ds.get_subject_ids()
    assert len(subject_ids) > 0
    assert all(isinstance(s, str) for s in subject_ids)
    assert subject_ids == sorted(subject_ids)


# ---------------------------------------------------------------------------
# Task 2: __getitem__ and helper methods
# ---------------------------------------------------------------------------


def test_getitem_schema(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    item = ds[0]
    required_keys = {"modality", "data", "label", "label_idx", "bert_emb", "subject_id", "subject_idx"}
    assert required_keys <= set(item.keys()), f"Missing keys: {required_keys - set(item.keys())}"


def test_getitem_data_is_tensor(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    item = ds[0]
    assert isinstance(item["data"], torch.Tensor)
    assert item["data"].dtype == torch.float32


def test_getitem_bert_emb_shape(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    item = ds[0]
    assert item["bert_emb"].shape == (768,), f"bert_emb shape: {item['bert_emb'].shape}"
    assert isinstance(item["bert_emb"], torch.Tensor)


def test_getitem_types(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    item = ds[0]
    assert isinstance(item["label"], str)
    assert isinstance(item["label_idx"], int)
    assert isinstance(item["subject_idx"], int)
    assert item["modality"] in ("eeg", "meg", "fmri")


def test_getitem_modality_coverage(stub_paths):
    """All three modalities should appear in the train split."""
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    modalities_seen = {ds[i]["modality"] for i in range(len(ds))}
    assert "eeg" in modalities_seen
    assert "meg" in modalities_seen
    assert "fmri" in modalities_seen


def test_helper_get_vocabulary(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    vocab = ds.get_vocabulary()
    assert isinstance(vocab, list)
    assert len(vocab) == 10


def test_helper_get_bert_embeddings(stub_paths):
    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    embs = ds.get_bert_embeddings()
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (10, 768)


def test_np_load_no_arbitrary_deserialization(stub_paths, monkeypatch):
    """Verify np.load is always called without allow_pickle=True."""
    import numpy as _np

    calls = []
    original_load = _np.load

    def patched_load(path, **kwargs):
        calls.append(kwargs.get("allow_pickle", False))
        return original_load(path, **kwargs)

    monkeypatch.setattr(_np, "load", patched_load)

    ds = MultiModalDataset(
        stub_paths["split_path"],
        stub_paths["vocab_path"],
        stub_paths["processed_dir"],
        split="train",
    )
    # Also trigger a getitem to cover lazy loading path
    _ = ds[0]

    assert all(v is False for v in calls), (
        f"np.load called with allow_pickle not False: {calls}"
    )
