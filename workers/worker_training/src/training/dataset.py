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
        processed_base_dir: str,
        split: str = "train",
        modalities: list = ("eeg", "meg", "fmri"),
    ):
        self.modalities = list(modalities)
        self.split = split

        with open(split_json_path, "r") as f:
            split_data = json.load(f)

        split_entry = split_data[split]

        vocab_npz = np.load(vocab_embeddings_path, allow_pickle=False)
        self.vocab = [str(w) for w in vocab_npz["vocab"]]
        self.bert_matrix = vocab_npz["embeddings"].astype(np.float32)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}

        self._index = []
        unique_subjects = set()

        for modality in [m for m in split_entry if m in self.modalities]:
            for filename in split_entry[modality]:
                path = Path(processed_base_dir) / modality / filename
                if not path.exists():
                    logger.warning("Missing file, skipping: %s", path)
                    continue
                npz = np.load(str(path), allow_pickle=False)
                n_rows = len(npz["labels"])
                subject_id = str(npz["subject_id"])
                unique_subjects.add(subject_id)
                for row_i in range(n_rows):
                    self._index.append((modality, path, row_i))

        self.subject2idx = {s: i for i, s in enumerate(sorted(unique_subjects))}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        modality, filepath, row_i = self._index[idx]
        npz = np.load(str(filepath), allow_pickle=False)

        data = npz["data"][row_i].astype(np.float32)
        label = str(npz["labels"][row_i])
        subject_id = str(npz["subject_id"])

        label_idx = self.word2idx.get(label)
        if label_idx is None:
            label_idx = 0

        bert_emb = torch.from_numpy(self.bert_matrix[label_idx].copy())

        return {
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
        result[modality] = {
            "data": torch.stack([item["data"] for item in items]),
            "label_idx": torch.tensor([item["label_idx"] for item in items], dtype=torch.long),
            "bert_emb": torch.stack([item["bert_emb"] for item in items]),
            "subject_idx": torch.tensor([item["subject_idx"] for item in items], dtype=torch.long),
            "labels": [item["label"] for item in items],
        }

    result["shared_label_mask"] = build_shared_label_mask(batch)
    return result
