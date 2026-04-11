"""
MultiModal dataset loader for Stage 1 and Stage 2 training.

Loads processed .npz files from all modalities, aligns vocabulary,
and provides batches with roughly equal samples per present modality.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
_INVALID_LABELS = {"", "nan", "none", "null"}


def _normalize_label(label) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, bytes):
        label = label.decode("utf-8", errors="ignore")
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, float) and np.isnan(label):
        return None

    text = str(label).strip()
    if text.lower() in _INVALID_LABELS:
        return None
    return text


def _resolve_split_entry(processed_base_dirs: list[str], modality: str, entry: str) -> Path:
    raw_path = Path(entry)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        for processed_base_dir in processed_base_dirs:
            base_dir = Path(processed_base_dir)
            candidates.extend(
                [
                    base_dir / raw_path,
                    base_dir / modality / raw_path,
                ]
            )
        candidates.append(raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


class MultiModalDataset(Dataset):
    """
    Dataset that samples from EEG, MEG, and/or fMRI processed .npz files.

    Each item is a dict:
        {
            'modality':    str
            'data':        Tensor
            'label':       str
            'label_idx':   int
            'bert_emb':    Tensor
            'subject_id':  str
            'subject_idx': int
        }
    """

    def __init__(
        self,
        split_json_path: str,
        vocab_embeddings_path: str,
        processed_base_dir: str | None,
        split: str = "train",
        modalities: list = ("eeg", "meg", "fmri"),
    ):
        self.modalities = list(modalities)
        self.split = split

        with open(split_json_path, "r") as f:
            split_data = json.load(f)

        split_entry = split_data[split]
        split_metadata = split_data.get("metadata", {})
        split_base_dir = split_metadata.get("processed_base_dir")
        self.processed_base_dir = processed_base_dir or split_base_dir or "data/processed"
        self._processed_base_dir_candidates = []
        for candidate in (processed_base_dir, split_base_dir, "data/processed"):
            if candidate and candidate not in self._processed_base_dir_candidates:
                self._processed_base_dir_candidates.append(candidate)

        vocab_npz = np.load(vocab_embeddings_path, allow_pickle=False)
        self.vocab = [str(w) for w in vocab_npz["vocab"]]
        self.bert_matrix = vocab_npz["embeddings"].astype(np.float32)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}

        self._index = []
        unique_subjects = set()

        for modality in [m for m in split_entry if m in self.modalities]:
            for entry in split_entry[modality]:
                path = _resolve_split_entry(self._processed_base_dir_candidates, modality, entry)
                if not path.exists():
                    logger.warning("Missing file, skipping: %s", path)
                    continue
                npz = np.load(str(path), allow_pickle=False)
                subject_id = str(npz["subject_id"])
                unique_subjects.add(subject_id)
                for row_i, raw_label in enumerate(npz["labels"].tolist()):
                    label = _normalize_label(raw_label)
                    if label is None or label not in self.word2idx:
                        continue
                    self._index.append((modality, path, row_i, label, self.word2idx[label]))

        self.subject2idx = {s: i for i, s in enumerate(sorted(unique_subjects))}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        modality, filepath, row_i, label, label_idx = self._index[idx]
        npz = np.load(str(filepath), allow_pickle=False)

        data = npz["data"][row_i].astype(np.float32)
        subject_id = str(npz["subject_id"])

        bert_emb = torch.from_numpy(self.bert_matrix[label_idx].copy())

        return {
            **({"sfreq": float(npz["sfreq"])} if "sfreq" in npz.files else {}),
            **({"voxel_coords": npz["voxel_coords"].astype(np.int32)} if "voxel_coords" in npz.files else {}),
            "modality": modality,
            "data": torch.from_numpy(data),
            "label": label,
            "label_idx": int(label_idx),
            "bert_emb": bert_emb,
            "subject_id": subject_id,
            "subject_idx": int(self.subject2idx[subject_id]),
        }

    def get_subject_ids(self):
        return sorted(self.subject2idx.keys())

    def get_vocabulary(self):
        return self.vocab

    def get_bert_embeddings(self):
        return self.bert_matrix


def build_shared_label_mask(batch):
    label_to_modalities = {}
    for sample in batch:
        label = sample["label"]
        modality = sample["modality"]
        if label not in label_to_modalities:
            label_to_modalities[label] = set()
        label_to_modalities[label].add(modality)

    mask = [len(label_to_modalities[sample["label"]]) > 1 for sample in batch]
    return torch.tensor(mask, dtype=torch.bool)


def collate_fn(batch):
    groups = {}
    for item in batch:
        modality = item["modality"]
        if modality not in groups:
            groups[modality] = []
        groups[modality].append(item)

    result = {}
    for modality, items in groups.items():
        modal_result = {
            "data": torch.stack([item["data"] for item in items]),
            "label_idx": torch.tensor([item["label_idx"] for item in items], dtype=torch.long),
            "bert_emb": torch.stack([item["bert_emb"] for item in items]),
            "subject_idx": torch.tensor([item["subject_idx"] for item in items], dtype=torch.long),
            "labels": [item["label"] for item in items],
        }
        if "sfreq" in items[0]:
            modal_result["sfreq"] = float(items[0]["sfreq"])
        if "voxel_coords" in items[0]:
            modal_result["voxel_coords"] = items[0]["voxel_coords"]
        result[modality] = modal_result

    result["shared_label_mask"] = build_shared_label_mask(batch)
    return result
