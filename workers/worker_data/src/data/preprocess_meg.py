"""
MEG preprocessing pipeline.

Same output schema as EEG, but with MEG-specific steps:
  - Picks both magnetometers and gradiometers
  - Applies Maxwell filtering (SSS/tSSS) for Elekta/MEGIN systems
  - Checks for movement artifacts if head position data is available
  - Per-channel z-score normalization handles magnetometer/gradiometer scale differences

Output schema (saved as .npz) — identical to EEG:
    data:           float32  (n_epochs, n_channels, n_timepoints)
    labels:         list[str]  length n_epochs
    subject_id:     str
    sfreq:          float
    ch_names:       list[str]
    event_onsets_s: float64  (n_epochs,)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_raw(file_path: str, dataset_card: dict):
    """
    Load a raw MEG file.

    Args:
        file_path: Path to the raw MEG file.
        dataset_card: Parsed DATASET_CARD.md as a dict.

    Returns:
        mne.io.Raw object.
    """
    raise NotImplementedError


def apply_maxwell_filter(raw, crosstalk_file: str = None, fine_cal_file: str = None):
    """
    Apply Maxwell filtering (SSS or tSSS) for Elekta/MEGIN systems.

    If crosstalk_file and fine_cal_file are available, uses them.
    Otherwise applies tSSS mode. Logs whether SSS was applied and which mode.

    Args:
        raw: mne.io.Raw MEG object.
        crosstalk_file: Path to crosstalk file, or None.
        fine_cal_file: Path to fine calibration file, or None.

    Returns:
        Maxwell-filtered mne.io.Raw object.
    """
    raise NotImplementedError


def annotate_movement_artifacts(raw):
    """
    Annotate segments with excessive head movement (if head position data available).

    Args:
        raw: mne.io.Raw MEG object.

    Returns:
        Raw with movement artifact annotations added (or unchanged if no
        head position data available — logs a warning).
    """
    raise NotImplementedError


def preprocess_raw(raw, config: dict):
    """
    Apply MEG-specific preprocessing: Maxwell filter, ICA, filtering, resampling.

    Steps (in order):
        1. Pick MEG channels (magnetometers + gradiometers)
        2. Apply Maxwell filter if Elekta/MEGIN system detected
        3. Annotate movement artifacts if head position data available
        4. Bandpass filter
        5. Notch filter
        6. Resample
        7. Fit and apply ICA

    Args:
        raw: mne.io.Raw MEG object.
        config: Dict matching configs/preprocessing_meg.yaml schema.

    Returns:
        Cleaned mne.io.Raw object.
    """
    raise NotImplementedError


def extract_epochs(raw, dataset_card: dict, config: dict):
    """
    Extract word-level epochs from cleaned MEG data.

    Identical contract to EEG version.

    Args:
        raw: Cleaned mne.io.Raw MEG object.
        dataset_card: Parsed dataset card.
        config: Dict matching configs/preprocessing_meg.yaml schema.

    Returns:
        Tuple of (mne.Epochs, list[str] labels).

    Raises:
        ValueError: If labels are sentence-level rather than word-level.
    """
    raise NotImplementedError


def normalize(epochs_data: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each channel across all epochs.

    Per-channel normalization naturally handles magnetometer/gradiometer
    scale differences without explicit unit conversion.

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
    Save preprocessed MEG epochs to .npz.

    Output path: data/processed/<dataset>/meg/sub-<id>_epochs.npz

    Args:
        data: float32 (n_epochs, n_channels, n_timepoints).
        labels: list of word strings, length n_epochs.
        subject_id: Subject identifier string.
        sfreq: Sampling frequency.
        ch_names: MEG channel names.
        event_onsets_s: Epoch onset times in seconds.
        output_dir: data/processed/<dataset>/meg/

    Returns:
        Path to saved .npz file.
    """
    raise NotImplementedError


def run_subject(subject_id: str, dataset_name: str, config: dict) -> str:
    """
    Run full MEG preprocessing pipeline for one subject.

    Args:
        subject_id: Subject identifier.
        dataset_name: Dataset name.
        config: Parsed preprocessing_meg.yaml config.

    Returns:
        Path to saved .npz file.
    """
    raise NotImplementedError


def run_all_subjects(dataset_name: str, config_path: str) -> list[str]:
    """
    Run MEG preprocessing for all subjects specified in config.

    Args:
        dataset_name: Dataset name.
        config_path: Path to configs/preprocessing_meg.yaml.

    Returns:
        List of paths to saved .npz files.
    """
    raise NotImplementedError
