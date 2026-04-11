"""
Dataset inspection utilities.

Run this on every new dataset BEFORE writing any preprocessing code.
Produces a DATASET_CARD.md at data/processed/<dataset_name>/DATASET_CARD.md.
"""

import logging
import pathlib
from collections import Counter

logger = logging.getLogger(__name__)


def audit_dataset(root_path: str) -> dict:
    """
    Step 1: Directory and file audit.

    Args:
        root_path: Path to the raw dataset root directory.

    Returns:
        dict with keys: 'file_types' (Counter), 'total_size_mb' (float),
        'file_count' (int), 'tree' (str).
    """
    raise NotImplementedError


def identify_format(root_path: str) -> dict:
    """
    Step 2: Format identification.

    Inspects file extensions and directory structure to determine
    data format and loading library.

    Args:
        root_path: Path to the raw dataset root directory.

    Returns:
        dict with keys: 'format' (str), 'loading_library' (str),
        'is_bids' (bool), 'notes' (str).
    """
    raise NotImplementedError


def inspect_eeg_meg_signal(file_path: str) -> dict:
    """
    Step 3a: Signal inspection for EEG/MEG datasets.

    Args:
        file_path: Path to a single raw EEG/MEG file.

    Returns:
        dict with keys: 'sfreq' (float), 'n_channels' (int),
        'ch_names' (list[str]), 'ch_types' (set[str]),
        'duration_s' (float), 'has_stim_channel' (bool).
    """
    raise NotImplementedError


def inspect_fmri_signal(file_path: str) -> dict:
    """
    Step 3b: Signal inspection for fMRI datasets.

    Args:
        file_path: Path to a NIfTI BOLD file.

    Returns:
        dict with keys: 'shape' (tuple), 'voxel_size_mm' (tuple),
        'tr_s' (float), 'n_timepoints' (int).
    """
    raise NotImplementedError


def inspect_stimulus_labels(dataset_path: str, modality: str) -> dict:
    """
    Step 4: Stimulus/label inspection. CRITICAL — pipeline requires word-level labels.

    Args:
        dataset_path: Path to the raw dataset root.
        modality: One of 'eeg', 'meg', 'fmri'.

    Returns:
        dict with keys: 'label_location' (str), 'label_format' (str),
        'label_granularity' (str: 'word' | 'sentence' | 'phrase'),
        'timing_precision' (str: 'samples' | 'seconds'),
        'n_unique_labels' (int), 'example_labels' (list[str]).

    Raises:
        ValueError: If label granularity cannot be determined.
    """
    raise NotImplementedError


def inspect_subject_structure(dataset_path: str) -> dict:
    """
    Step 5: Subject and session structure.

    Args:
        dataset_path: Path to the raw dataset root.

    Returns:
        dict with keys: 'n_subjects' (int), 'subject_ids' (list[str]),
        'n_sessions_per_subject' (dict), 'has_held_out_test' (bool).
    """
    raise NotImplementedError


def write_dataset_card(
    dataset_name: str,
    modality: str,
    format_info: dict,
    signal_info: dict,
    subject_info: dict,
    label_info: dict,
    known_issues: list[str],
    output_dir: str,
) -> str:
    """
    Step 6: Write DATASET_CARD.md after all inspection steps complete.

    Args:
        dataset_name: Short identifier used in file paths.
        modality: One of 'eeg', 'meg', 'fmri'.
        format_info: Output of identify_format().
        signal_info: Output of inspect_eeg_meg_signal() or inspect_fmri_signal().
        subject_info: Output of inspect_subject_structure().
        label_info: Output of inspect_stimulus_labels().
        known_issues: List of known quirks, missing data, bad channels, etc.
        output_dir: Directory to write the card (typically data/processed/<dataset_name>/).

    Returns:
        Path to the written DATASET_CARD.md.
    """
    raise NotImplementedError


def run_full_inspection(dataset_name: str, root_path: str, modality: str) -> str:
    """
    Run all 6 inspection steps and write the dataset card.

    This is the entry point. Call this before writing any preprocessing code
    for a new dataset.

    Args:
        dataset_name: Short identifier (e.g., 'brennan2019').
        root_path: Path to data/raw/<dataset_name>/.
        modality: One of 'eeg', 'meg', 'fmri'.

    Returns:
        Path to the written DATASET_CARD.md.
    """
    raise NotImplementedError
