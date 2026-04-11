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
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError

    def get_subject_ids(self) -> list[str]:
        """Return sorted list of all subject IDs in this split."""
        raise NotImplementedError

    def get_vocabulary(self) -> list[str]:
        """Return the full vocabulary list."""
        raise NotImplementedError

    def get_bert_embeddings(self) -> np.ndarray:
        """Return (V, 768) BERT embedding matrix for the full vocabulary."""
        raise NotImplementedError


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
    raise NotImplementedError
