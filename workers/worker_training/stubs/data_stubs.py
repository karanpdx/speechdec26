"""
Data stubs for worker_training development.

Generates synthetic data matching the exact schemas produced by worker_data.
Use these while Person 1 builds the real preprocessing pipeline.

At integration, swap the data directory paths to point to real processed data.
The schemas are identical — no code changes needed in training modules.
"""

import json
import tempfile
from pathlib import Path

import numpy as np


def make_eeg_npz(
    n_epochs: int = 100,
    n_channels: int = 64,
    n_timepoints: int = 175,
    n_words: int = 20,
    subject_id: str = "sub-01",
    sfreq: float = 256.0,
) -> dict:
    """
    Synthetic EEG .npz matching the output schema from preprocess_eeg.py.

    Returns:
        Dict matching the .npz schema (can be passed directly or saved with np.savez).
    """
    data = np.random.randn(n_epochs, n_channels, n_timepoints).astype(np.float32)
    data = (data - data.mean(axis=(0, 2), keepdims=True)) / (data.std(axis=(0, 2), keepdims=True) + 1e-8)
    vocab = [f"word_{i:03d}" for i in range(n_words)]
    labels = [vocab[i % n_words] for i in range(n_epochs)]
    return {
        "data": data,
        "labels": np.array(labels),
        "subject_id": subject_id,
        "sfreq": np.float64(sfreq),
        "ch_names": np.array([f"EEG{i:03d}" for i in range(n_channels)]),
        "event_onsets_s": np.linspace(0, n_epochs * 0.8, n_epochs),
    }


def make_fmri_npz(
    n_words: int = 20,
    n_voxels: int = 1000,
    subject_id: str = "sub-01",
) -> dict:
    """
    Synthetic fMRI .npz matching the output schema from preprocess_fmri.py.
    """
    data = np.random.randn(n_words, n_voxels).astype(np.float32)
    labels = [f"word_{i:03d}" for i in range(n_words)]
    voxel_coords = np.random.randint(-90, 90, size=(n_voxels, 3), dtype=np.int32)
    return {
        "data": data,
        "labels": np.array(labels),
        "subject_id": subject_id,
        "voxel_coords": voxel_coords,
        "pca_explained_variance": np.float64(0.0),
    }


def make_vocab_embeddings(n_words: int = 20, embed_dim: int = 768) -> dict:
    """
    Synthetic vocab embeddings matching data/processed/vocab_embeddings.npz.
    """
    vocab = [f"word_{i:03d}" for i in range(n_words)]
    embeddings = np.random.randn(n_words, embed_dim).astype(np.float32)
    # Normalize to unit sphere (BERT embeddings are approximately normalized)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return {
        "vocab": np.array(vocab),
        "embeddings": embeddings,
    }


def make_split_json(
    subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
    modalities: list[str] = ("eeg", "meg", "fmri"),
) -> dict:
    """
    Synthetic split index matching data/splits/split_v1.json schema.
    """
    train_subjects = [s for s in subjects if s not in val_subjects and s not in test_subjects]

    def files_for(subs, modality):
        return [f"{s}_epochs.npz" if modality != "fmri" else f"{s}_betas.npz" for s in subs]

    return {
        "train": {m: files_for(train_subjects, m) for m in modalities},
        "val":   {m: files_for(val_subjects, m) for m in modalities},
        "test":  {m: files_for(test_subjects, m) for m in modalities},
    }


def write_stub_dataset(
    base_dir: str,
    n_subjects: int = 8,
    n_epochs_per_subject: int = 80,
    n_channels_eeg: int = 64,
    n_channels_meg: int = 306,
    n_voxels: int = 1000,
    n_timepoints: int = 175,
    n_words: int = 20,
    modalities: list[str] = ("eeg", "meg", "fmri"),
) -> dict:
    """
    Write a complete synthetic dataset to disk at base_dir.

    Creates the directory structure expected by MultiModalDataset:
        base_dir/
            processed/
                stub_dataset/eeg/sub-XX_epochs.npz
                stub_dataset/meg/sub-XX_epochs.npz
                stub_dataset/fmri/sub-XX_betas.npz
                vocab_embeddings.npz
            splits/split_v1.json

    Returns:
        Dict with paths: {'split_path', 'vocab_path', 'processed_dir'}
    """
    base = Path(base_dir)
    subjects = [f"sub-{i:02d}" for i in range(1, n_subjects + 1)]
    val_subjects = subjects[-2:-1]
    test_subjects = subjects[-1:]

    for modality in modalities:
        modal_dir = base / "processed" / "stub_dataset" / modality
        modal_dir.mkdir(parents=True, exist_ok=True)
        for subj in subjects:
            if modality in ("eeg", "meg"):
                n_ch = n_channels_eeg if modality == "eeg" else n_channels_meg
                npz = make_eeg_npz(n_epochs=n_epochs_per_subject, n_channels=n_ch,
                                    n_timepoints=n_timepoints, n_words=n_words,
                                    subject_id=subj)
                np.savez(modal_dir / f"{subj}_epochs.npz", **npz)
            else:
                npz = make_fmri_npz(n_words=n_words, n_voxels=n_voxels, subject_id=subj)
                np.savez(modal_dir / f"{subj}_betas.npz", **npz)

    vocab_dir = base / "processed"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_data = make_vocab_embeddings(n_words=n_words)
    np.savez(vocab_dir / "vocab_embeddings.npz", **vocab_data)

    splits_dir = base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_data = make_split_json(subjects, val_subjects, test_subjects, modalities)
    split_path = splits_dir / "split_v1.json"
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)

    return {
        "split_path": str(split_path),
        "vocab_path": str(vocab_dir / "vocab_embeddings.npz"),
        "processed_dir": str(base / "processed" / "stub_dataset"),
    }
