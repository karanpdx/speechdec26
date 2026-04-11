"""
Dataset alignment and split generation.

Aligns processed outputs across modalities, ensures consistent vocabulary,
and generates reproducible train/val/test splits stratified by subject.

Invariants (verified before saving):
  - No test subject appears in train or val
  - No val subject appears in train
  - Vocabulary is identical across all splits
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_vocabulary(processed_dirs: dict[str, str]) -> list[str]:
    """
    Build the union of all unique word labels across all processed .npz files.

    Args:
        processed_dirs: Dict mapping modality name to path of processed directory.
                        e.g. {'eeg': 'data/processed/dataset/eeg/', ...}

    Returns:
        Sorted list of unique word strings.
    """
    raise NotImplementedError


def filter_vocabulary(
    vocabulary: list[str],
    processed_dirs: dict[str, str],
    train_subject_ids: list[str],
    min_word_freq: int,
    vocab_source: str,
) -> list[str]:
    """
    Filter vocabulary by minimum frequency in training subjects.

    Args:
        vocabulary: Full vocabulary list.
        processed_dirs: Dict of modality → processed directory path.
        train_subject_ids: Subject IDs in the training set.
        min_word_freq: Minimum occurrences across all training modalities combined.
        vocab_source: 'intersection' (word must appear in all modalities)
                      or 'union' (word must appear in any modality).

    Returns:
        Filtered vocabulary list.
    """
    raise NotImplementedError


def generate_splits(
    processed_dirs: dict[str, str],
    val_subjects: list[str],
    test_subjects: list[str],
    vocabulary: list[str],
) -> dict:
    """
    Generate train/val/test splits as a dict of subject file lists per modality.

    Args:
        processed_dirs: Dict of modality → processed directory path.
        val_subjects: Subject IDs held out for validation.
        test_subjects: Subject IDs held out for test.
        vocabulary: Filtered vocabulary (only files with labels in vocab are included).

    Returns:
        Dict: {'train': {'eeg': [...], 'meg': [...], 'fmri': [...]},
               'val':   {'eeg': [...], ...},
               'test':  {'eeg': [...], ...}}
    """
    raise NotImplementedError


def verify_split_integrity(splits: dict, val_subjects: list[str], test_subjects: list[str]) -> None:
    """
    Assert that no subject leakage exists between splits.

    Args:
        splits: Output of generate_splits().
        val_subjects: Expected val subject IDs.
        test_subjects: Expected test subject IDs.

    Raises:
        AssertionError: If any subject appears in more than one split.
    """
    raise NotImplementedError


def print_split_statistics(splits: dict, vocabulary: list[str]) -> None:
    """
    Print split statistics to logs: subjects, epoch/word counts,
    vocabulary size, per-modality counts.

    Logs a warning for any modality with zero samples in a split.

    Args:
        splits: Output of generate_splits().
        vocabulary: Filtered vocabulary list.
    """
    raise NotImplementedError


def save_splits(splits: dict, output_path: str) -> str:
    """
    Save splits to JSON.

    Args:
        splits: Output of generate_splits().
        output_path: Path for split JSON (e.g., 'data/splits/split_v1.json').

    Returns:
        Path to saved JSON file.
    """
    raise NotImplementedError


def run_alignment(config_path: str) -> tuple[str, str]:
    """
    Run full dataset alignment and split generation.

    Entry point called by the alignment script.

    Args:
        config_path: Path to configs/splits.yaml.

    Returns:
        Tuple of (split_json_path, vocab_embeddings_path).
    """
    raise NotImplementedError
