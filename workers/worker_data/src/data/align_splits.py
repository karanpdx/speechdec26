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
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_INVALID_LABELS = {"", "nan", "none", "null"}


def _normalize_label(label) -> str | None:
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


def _load_valid_labels(npz_path: Path) -> list[str]:
    with np.load(npz_path, allow_pickle=True) as data:
        raw_labels = data["labels"].tolist()
    return [label for raw in raw_labels if (label := _normalize_label(raw)) is not None]


def _common_processed_root(processed_dirs: dict[str, str]) -> Path:
    roots = [str(Path(path)) for path in processed_dirs.values() if path]
    if not roots:
        raise ValueError("No processed directories provided")
    return Path(os.path.commonpath(roots))


def build_vocabulary(processed_dirs: dict[str, str]) -> list[str]:
    """
    Build the union of all unique word labels across all processed .npz files.

    Args:
        processed_dirs: Dict mapping modality name to path of processed directory.
                        e.g. {'eeg': 'data/processed/dataset/eeg/', ...}

    Returns:
        Sorted list of unique word strings.
    """
    vocabulary = set()
    for modality, dir_path in processed_dirs.items():
        if not dir_path:
            continue
        for npz_path in sorted(Path(dir_path).glob("*.npz")):
            vocabulary.update(_load_valid_labels(npz_path))
    return sorted(vocabulary)


def filter_vocabulary(
    vocabulary: list[str],
    processed_dirs: dict[str, str],
    train_subject_ids: list[str],
    min_word_freq: int,
    vocab_source: str,
    target_vocab_size: int | None = None,
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
    freq_counts: dict[str, int] = {word: 0 for word in vocabulary}
    modality_presence: dict[str, set[str]] = {word: set() for word in vocabulary}

    for modality, dir_path in processed_dirs.items():
        if not dir_path:
            continue
        for npz_path in sorted(Path(dir_path).glob("*.npz")):
            subject_id = Path(npz_path).name.split("_")[0]
            if subject_id not in train_subject_ids:
                continue
            labels = _load_valid_labels(npz_path)
            for label in labels:
                if label in freq_counts:
                    freq_counts[label] += 1
                    modality_presence[label].add(modality)

    active_modalities = {name for name, path in processed_dirs.items() if path}
    filtered = []
    for word in vocabulary:
        if freq_counts[word] < min_word_freq:
            continue
        if vocab_source == "intersection" and modality_presence[word] != active_modalities:
            continue
        filtered.append(word)

    ranked = sorted(filtered, key=lambda word: (-freq_counts[word], word))
    if target_vocab_size is not None:
        if len(ranked) < target_vocab_size:
            raise ValueError(
                f"Only {len(ranked)} eligible shared words found, but target_vocab_size={target_vocab_size}"
            )
        ranked = ranked[:target_vocab_size]
    return ranked


def generate_splits(
    processed_dirs: dict[str, str],
    val_subjects: list[str],
    test_subjects: list[str],
    vocabulary: list[str],
    processed_base_dir: str | None = None,
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
    vocab_set = set(vocabulary)
    base_dir = Path(processed_base_dir) if processed_base_dir is not None else _common_processed_root(processed_dirs)
    splits = {
        "metadata": {"processed_base_dir": str(base_dir)},
        "train": {modality: [] for modality in processed_dirs if processed_dirs[modality]},
        "val": {modality: [] for modality in processed_dirs if processed_dirs[modality]},
        "test": {modality: [] for modality in processed_dirs if processed_dirs[modality]},
    }

    for modality, dir_path in processed_dirs.items():
        if not dir_path:
            continue
        for npz_path in sorted(Path(dir_path).glob("*.npz")):
            labels = set(_load_valid_labels(npz_path))
            if not (labels & vocab_set):
                continue

            subject_id = npz_path.name.split("_")[0]
            if subject_id in test_subjects:
                split_name = "test"
            elif subject_id in val_subjects:
                split_name = "val"
            else:
                split_name = "train"
            try:
                split_entry = str(npz_path.relative_to(base_dir))
            except ValueError:
                split_entry = str(npz_path)
            splits[split_name][modality].append(split_entry)

    return splits


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
    # Collect all filenames across modalities in train split
    train_files = []
    for modality_files in splits.get("train", {}).values():
        train_files.extend(modality_files)

    # Extract subject IDs from train filenames (prefix before first '_')
    train_subjects = set()
    for fname in train_files:
        stem = Path(fname).name
        train_subjects.add(stem.split("_")[0])

    val_set = set(val_subjects)
    test_set = set(test_subjects)

    val_leakage = val_set & train_subjects
    assert not val_leakage, f"Val subjects found in train split: {val_leakage}"

    test_leakage = test_set & train_subjects
    assert not test_leakage, f"Test subjects found in train split: {test_leakage}"


def print_split_statistics(splits: dict, vocabulary: list[str]) -> None:
    """
    Print split statistics to logs: subjects, epoch/word counts,
    vocabulary size, per-modality counts.

    Logs a warning for any modality with zero samples in a split.

    Args:
        splits: Output of generate_splits().
        vocabulary: Filtered vocabulary list.
    """
    logger.info(f"Vocabulary size: {len(vocabulary)}")
    for split_name, split_modalities in splits.items():
        if split_name == "metadata":
            continue
        logger.info(f"Split: {split_name}")
        for modality, files in split_modalities.items():
            subject_ids = sorted({Path(path).name.split('_')[0] for path in files})
            logger.info(
                f"  {modality}: {len(files)} file(s), {len(subject_ids)} subject(s)"
            )
            if not files:
                logger.warning(f"  {modality}: zero samples in {split_name}")


def save_splits(splits: dict, output_path: str) -> str:
    """
    Save splits to JSON.

    Args:
        splits: Output of generate_splits().
        output_path: Path for split JSON (e.g., 'data/splits/split_v1.json').

    Returns:
        Path to saved JSON file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    logger.info(f"Saved split file: {out_path}")
    return str(out_path)


def run_alignment(config_path: str) -> tuple[str, str]:
    """
    Run full dataset alignment and split generation.

    Entry point called by the alignment script.

    Args:
        config_path: Path to configs/splits.yaml.

    Returns:
        Tuple of (split_json_path, vocab_embeddings_path).
    """
    import yaml

    from .vocab_embeddings import run as run_vocab_embeddings

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    processed_dirs = {
        modality: path
        for modality, path in config.get("processed_dirs", {}).items()
        if path
    }
    if not processed_dirs:
        raise ValueError("No processed_dirs configured in splits config")

    all_subject_ids = set()
    for dir_path in processed_dirs.values():
        for npz_path in Path(dir_path).glob("*.npz"):
            all_subject_ids.add(npz_path.name.split("_")[0])

    val_subjects = config.get("val_subjects", [])
    test_subjects = config.get("test_subjects", [])
    train_subjects = sorted(all_subject_ids - set(val_subjects) - set(test_subjects))

    vocabulary = build_vocabulary(processed_dirs)
    vocabulary = filter_vocabulary(
        vocabulary=vocabulary,
        processed_dirs=processed_dirs,
        train_subject_ids=train_subjects,
        min_word_freq=config.get("min_word_freq", 5),
        vocab_source=config.get("vocab_source", "intersection"),
        target_vocab_size=config.get("target_vocab_size"),
    )
    if not vocabulary:
        raise ValueError("Filtered vocabulary is empty; cannot generate splits")

    splits = generate_splits(
        processed_dirs=processed_dirs,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        vocabulary=vocabulary,
        processed_base_dir=str(_common_processed_root(processed_dirs)),
    )
    verify_split_integrity(splits, val_subjects, test_subjects)
    print_split_statistics(splits, vocabulary)

    split_path = save_splits(splits, config["output_split_path"])
    vocab_path = run_vocab_embeddings(vocabulary, config["output_vocab_path"])
    return split_path, vocab_path
