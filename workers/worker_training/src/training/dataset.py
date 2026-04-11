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
            'modality':    str — 'eeg', 'meg', or 'fmri'
            'data':        Tensor — (n_channels, n_timepoints) or (n_voxels,)
            'label':       str — word string
            'label_idx':   int — index into vocabulary
            'bert_emb':    Tensor — (768,) BERT embedding for this word
            'subject_id':  str
            'subject_idx': int — index for SubjectEmbedding lookup
        }

    Batches contain samples from all available modalities, roughly balanced.
    Missing modalities for a given batch do not cause errors.
    """

    def __init__(
        self,
        split_json_path: str,
        vocab_embeddings_path: str,
        processed_base_dir: str,
        split: str = "train",
        modalities: list[str] = ("eeg", "meg", "fmri"),
    ):
        """
        Args:
            split_json_path: Path to data/splits/split_v1.json.
            vocab_embeddings_path: Path to data/processed/vocab_embeddings.npz.
            processed_base_dir: Base directory containing modality subdirs.
            split: One of 'train', 'val', 'test'.
            modalities: Which modalities to include.
        """
        self.modalities = list(modalities)
        self.split = split

        # Load split JSON
        with open(split_json_path, "r") as f:
            split_data = json.load(f)

        split_entry = split_data[split]  # {modality: [filename, ...]}

        # Load vocabulary and BERT embeddings — allow_pickle=False prevents
        # deserialization of arbitrary objects from untrusted .npz files.
        vocab_npz = np.load(vocab_embeddings_path, allow_pickle=False)
        self.vocab: list[str] = [str(w) for w in vocab_npz["vocab"]]
        self.bert_matrix: np.ndarray = vocab_npz["embeddings"].astype(np.float32)
        self.word2idx: dict[str, int] = {w: i for i, w in enumerate(self.vocab)}
        logger.info(
            "Loaded vocabulary: %d words, bert_matrix shape: %s",
            len(self.vocab),
            self.bert_matrix.shape,
        )

        # Build flat index: list of (modality, filepath, row_i)
        # Data arrays are NOT loaded into RAM here — lazy loading happens in __getitem__.
        self._index: list[tuple[str, Path, int]] = []
        unique_subjects: set[str] = set()

        for modality in [m for m in split_entry if m in self.modalities]:
            for filename in split_entry[modality]:
                path = Path(processed_base_dir) / modality / filename
                if not path.exists():
                    logger.warning("Missing file, skipping: %s", path)
                    continue
                # Open briefly to count rows and collect subject_id; no data retained.
                npz = np.load(str(path), allow_pickle=False)
                n_rows = len(npz["labels"])
                subject_id = str(npz["subject_id"])
                unique_subjects.add(subject_id)
                for row_i in range(n_rows):
                    self._index.append((modality, path, row_i))
                logger.info(
                    "Indexed %s/%s: %d samples (subject=%s)",
                    modality,
                    filename,
                    n_rows,
                    subject_id,
                )

        self.subject2idx: dict[str, int] = {
            s: i for i, s in enumerate(sorted(unique_subjects))
        }
        logger.info(
            "Split=%s — total samples: %d, subjects: %d, modalities: %s",
            split,
            len(self._index),
            len(self.subject2idx),
            self.modalities,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        modality, filepath, row_i = self._index[idx]
        # allow_pickle=False: treat .npz data arrays as untrusted filesystem input.
        npz = np.load(str(filepath), allow_pickle=False)

        data = npz["data"][row_i].astype(np.float32)
        label = str(npz["labels"][row_i])
        subject_id = str(npz["subject_id"])

        label_idx = self.word2idx.get(label)
        if label_idx is None:
            logger.warning("Unknown label '%s' at index %d, mapping to index 0", label, idx)
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

    def get_subject_ids(self) -> list[str]:
        """Return sorted list of all subject IDs in this split."""
        return sorted(self.subject2idx.keys())

    def get_vocabulary(self) -> list[str]:
        """Return the full vocabulary list."""
        return self.vocab

    def get_bert_embeddings(self) -> np.ndarray:
        """Return (V, 768) BERT embedding matrix for the full vocabulary."""
        return self.bert_matrix


def build_shared_label_mask(batch: list[dict]) -> torch.Tensor:
    """
    Build a boolean mask for CrossModalAlignmentLoss.

    A sample gets True if the same word label appears in at least
    one other sample from a different modality in this batch.

    Args:
        batch: List of item dicts from MultiModalDataset.

    Returns:
        (batch_size,) bool tensor.
    """
    label_to_modalities: dict[str, set[str]] = {}
    for sample in batch:
        label = sample["label"]
        modality = sample["modality"]
        if label not in label_to_modalities:
            label_to_modalities[label] = set()
        label_to_modalities[label].add(modality)

    mask = [len(label_to_modalities[sample["label"]]) > 1 for sample in batch]
    return torch.tensor(mask, dtype=torch.bool)


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for MultiModalDataset.

    Groups samples by modality and stacks tensors.

    Returns:
        Dict with keys per modality:
        {
            'eeg': {'data': Tensor, 'label_idx': Tensor, 'bert_emb': Tensor,
                    'subject_idx': Tensor, 'labels': list[str]},
            'meg': { ... },
            'fmri': { ... },
            'shared_label_mask': BoolTensor (batch_size,)
        }
        Modalities with no samples in this batch are absent from the dict.
    """
    # Group items by modality
    groups: dict[str, list[dict]] = {}
    for item in batch:
        modality = item["modality"]
        if modality not in groups:
            groups[modality] = []
        groups[modality].append(item)

    result: dict = {}
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
