"""
End-to-end tests for the dataset loader.

Uses write_stub_dataset to generate synthetic data in a temp dir — no real data needed.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stubs.data_stubs import write_stub_dataset
from src.training.dataset import MultiModalDataset, collate_fn, build_shared_label_mask


class TestMultiModalDataset:

    def test_instantiation(self, tmp_path):
        """Dataset instantiates and has non-zero length."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        assert len(ds) > 0

    def test_item_schema(self, tmp_path):
        """__getitem__ returns dict with all required keys and correct types."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        item = ds[0]
        assert {
            "modality", "data", "label", "label_idx", "bert_emb", "subject_id", "subject_idx"
        }.issubset(set(item.keys()))
        assert isinstance(item["data"], torch.Tensor)
        assert item["bert_emb"].shape == (768,)
        assert item["data"].dtype == torch.float32
        assert item["modality"] in ("eeg", "meg", "fmri")
        assert isinstance(item["label"], str)
        assert isinstance(item["label_idx"], int)
        assert isinstance(item["subject_idx"], int)

    def test_eeg_data_shape(self, tmp_path):
        """EEG items have shape (n_channels, n_timepoints)."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        eeg_items = [ds[i] for i in range(len(ds)) if ds[i]["modality"] == "eeg"]
        assert len(eeg_items) > 0
        assert eeg_items[0]["data"].shape == (64, 175)

    def test_fmri_data_shape(self, tmp_path):
        """fMRI items have shape (n_voxels,)."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        fmri_items = [ds[i] for i in range(len(ds)) if ds[i]["modality"] == "fmri"]
        assert len(fmri_items) > 0
        assert fmri_items[0]["data"].shape == (1000,)

    def test_missing_modality_skipped(self, tmp_path):
        """Dataset instantiates with only one modality requested."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"],
                                split="train", modalities=["eeg"])
        assert len(ds) > 0
        assert all(ds[i]["modality"] == "eeg" for i in range(min(10, len(ds))))

    def test_vocab_helpers(self, tmp_path):
        """Helper methods return correct types and sizes."""
        paths = write_stub_dataset(str(tmp_path), n_words=20)
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        assert isinstance(ds.get_vocabulary(), list)
        assert len(ds.get_vocabulary()) == 20
        assert ds.get_bert_embeddings().shape == (20, 768)
        assert isinstance(ds.get_subject_ids(), list)
        assert len(ds.get_subject_ids()) > 0

    def test_relative_split_entries_resolve_from_split_metadata(self, tmp_path):
        """Dataset resolves relative split entries against metadata processed_base_dir."""
        paths = write_stub_dataset(str(tmp_path))
        split_path = Path(paths["split_path"])
        with open(split_path, "r", encoding="utf-8") as handle:
            split_data = json.load(handle)

        split_data["metadata"] = {"processed_base_dir": paths["processed_dir"]}
        for split_name in ("train", "val", "test"):
            for modality, entries in split_data[split_name].items():
                split_data[split_name][modality] = [f"{modality}/{entry}" for entry in entries]

        with open(split_path, "w", encoding="utf-8") as handle:
            json.dump(split_data, handle)

        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], None, split="train")
        assert len(ds) > 0

    def test_split_metadata_fallback_works_when_config_base_dir_is_generic(self, tmp_path):
        """Metadata base dir should still resolve files when config points at data/processed."""
        paths = write_stub_dataset(str(tmp_path))
        split_path = Path(paths["split_path"])
        with open(split_path, "r", encoding="utf-8") as handle:
            split_data = json.load(handle)

        split_data["metadata"] = {"processed_base_dir": paths["processed_dir"]}
        for split_name in ("train", "val", "test"):
            for modality, entries in split_data[split_name].items():
                split_data[split_name][modality] = [f"{modality}/{entry}" for entry in entries]

        with open(split_path, "w", encoding="utf-8") as handle:
            json.dump(split_data, handle)

        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], "data/processed", split="train")
        assert len(ds) > 0


class TestCollateFn:

    def test_dataloader_iterates_three_batches(self, tmp_path):
        """DataLoader with collate_fn iterates 3 batches without raising."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn, shuffle=False)
        count = 0
        for batch in loader:
            count += 1
            if count >= 3:
                break
        assert count >= 3 or count == len(loader), "Expected at least 3 batches or full dataset"

    def test_batch_modality_tensor_shapes(self, tmp_path):
        """Tensors in each modality sub-dict have correct shapes."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        loader = DataLoader(ds, batch_size=24, collate_fn=collate_fn, shuffle=False)
        batch = next(iter(loader))
        if "eeg" in batch:
            assert batch["eeg"]["data"].ndim == 3          # (N, 64, 175)
            assert batch["eeg"]["data"].shape[-2] == 64
            assert batch["eeg"]["data"].shape[-1] == 175
            assert batch["eeg"]["bert_emb"].shape[-1] == 768
            assert batch["eeg"]["label_idx"].dtype == torch.long
            assert "sfreq" in batch["eeg"]
        if "meg" in batch:
            assert batch["meg"]["data"].shape[-2] == 306
            assert "sfreq" in batch["meg"]
        if "fmri" in batch:
            assert batch["fmri"]["data"].ndim == 2          # (N, 1000)
            assert batch["fmri"]["data"].shape[-1] == 1000
            assert "voxel_coords" in batch["fmri"]

    def test_shared_label_mask_in_batch(self, tmp_path):
        """collate_fn output always has shared_label_mask as a BoolTensor."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn, shuffle=False)
        batch = next(iter(loader))
        assert "shared_label_mask" in batch
        assert batch["shared_label_mask"].dtype == torch.bool
        assert batch["shared_label_mask"].shape == (16,) or batch["shared_label_mask"].shape[0] <= 16


class TestBuildSharedLabelMask:

    def test_cross_modal_match_is_true(self):
        """Labels shared across modalities are marked True."""
        batch = [
            {"modality": "eeg",  "label": "cat"},
            {"modality": "meg",  "label": "cat"},
            {"modality": "eeg",  "label": "dog"},
        ]
        mask = build_shared_label_mask(batch)
        assert mask.tolist() == [True, True, False]

    def test_same_modality_only_is_false(self):
        """Labels that only appear within one modality are marked False."""
        batch = [
            {"modality": "eeg", "label": "cat"},
            {"modality": "eeg", "label": "cat"},
        ]
        mask = build_shared_label_mask(batch)
        assert mask.tolist() == [False, False]

    def test_mask_shape_matches_batch_size(self, tmp_path):
        """Mask length equals total number of samples in the batch."""
        paths = write_stub_dataset(str(tmp_path))
        ds = MultiModalDataset(paths["split_path"], paths["vocab_path"], paths["processed_dir"], split="train")
        batch_list = [ds[i] for i in range(12)]
        mask = build_shared_label_mask(batch_list)
        assert mask.shape == (12,)
        assert mask.dtype == torch.bool
