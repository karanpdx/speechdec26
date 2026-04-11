"""
EEG preprocessing pipeline.

Transforms raw EEG recordings into clean, epoched, normalized tensors.

Output schema (saved as .npz):
    data:           float32  (n_epochs, n_channels, n_timepoints)
    labels:         list[str]  length n_epochs
    subject_id:     str
    sfreq:          float
    ch_names:       list[str]
    event_onsets_s: float64  (n_epochs,)
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_raw(file_path: str, dataset_card: dict):
    """
    Load a raw EEG file using the loader identified in the dataset card.

    Args:
        file_path: Path to the raw EEG file.
        dataset_card: Parsed DATASET_CARD.md as a dict (format, loading_library, etc.).

    Returns:
        mne.io.Raw object with data not yet loaded into memory.
    """
    raise NotImplementedError


def preprocess_raw(raw, config: dict):
    """
    Apply filtering, resampling, and ICA artifact removal.

    Steps (in order):
        1. Pick EEG channels only
        2. Set montage (standard 10-20 if unknown)
        3. Bandpass filter (l_freq, h_freq, method='firwin')
        4. Notch filter at specified frequencies
        5. Resample to target_sfreq if needed
        6. Fit ICA, auto-identify and remove EOG/ECG components
        7. Apply ICA to raw

    Args:
        raw: mne.io.Raw object.
        config: Dict matching configs/preprocessing_eeg.yaml schema.

    Returns:
        Cleaned mne.io.Raw object.
    """
    raise NotImplementedError


def extract_epochs(raw, dataset_card: dict, config: dict):
    """
    Extract word-level epochs from cleaned raw data.

    Args:
        raw: Cleaned mne.io.Raw object.
        dataset_card: Parsed dataset card (label location, format, etc.).
        config: Dict matching configs/preprocessing_eeg.yaml schema.

    Returns:
        Tuple of (mne.Epochs, list[str] labels).

    Raises:
        ValueError: If labels are sentence-level rather than word-level.
    """
    raise NotImplementedError


def normalize(epochs_data: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each channel across all epochs.

    Mean and std computed across time and epoch dimensions per channel.

    Args:
        epochs_data: float32 array (n_epochs, n_channels, n_timepoints).

    Returns:
        Normalized float32 array, same shape.
    """
    raise NotImplementedError


def save_epochs(
    data: np.ndarray,
    labels: list,
    subject_id: str,
    sfreq: float,
    ch_names: list,
    event_onsets_s: np.ndarray,
    output_dir: str,
) -> str:
    """
    Save preprocessed epochs to .npz.

    Runs output validation assertions before saving.

    Args:
        data: float32 (n_epochs, n_channels, n_timepoints).
        labels: list of word strings, length n_epochs.
        subject_id: Subject identifier string.
        sfreq: Sampling frequency after resampling.
        ch_names: Channel names after picking.
        event_onsets_s: Epoch onset times in seconds.
        output_dir: Directory to write output (data/processed/<dataset>/eeg/).

    Returns:
        Path to saved .npz file.

    Raises:
        AssertionError: If output validation fails.
    """
    raise NotImplementedError


def run_subject(subject_id: str, dataset_name: str, config: dict) -> str:
    """
    Run full EEG preprocessing pipeline for one subject.

    Entry point called by the preprocessing script.

    Args:
        subject_id: Subject identifier (e.g., 'sub-01').
        dataset_name: Dataset name matching data/raw/<dataset_name>/.
        config: Parsed preprocessing_eeg.yaml config.

    Returns:
        Path to saved .npz file.
    """
    raise NotImplementedError


def run_all_subjects(dataset_name: str, config_path: str) -> list[str]:
    """
    Run EEG preprocessing for all subjects specified in config.

    Args:
        dataset_name: Dataset name.
        config_path: Path to configs/preprocessing_eeg.yaml.

    Returns:
        List of paths to saved .npz files.
    """
    raise NotImplementedError
