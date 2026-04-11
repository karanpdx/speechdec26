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
        file_path: Path to the raw EEG file (.vhdr for BrainVision).
        dataset_card: Parsed DATASET_CARD.md as a dict (format, loading_library, etc.).

    Returns:
        mne.io.Raw object with data not yet loaded into memory.
    """
    import mne

    logger.info(f"Loading BrainVision EEG: {file_path}")
    raw = mne.io.read_raw_brainvision(str(file_path), preload=False, verbose=False)
    return raw


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
    from mne.preprocessing import ICA

    raw.load_data(verbose=False)
    raw.pick_types(eeg=True, exclude="bads")
    logger.info(f"EEG channels: {len(raw.ch_names)}, sfreq={raw.info['sfreq']} Hz")

    # Try to set montage — ZuCo uses Geodesic Sensor Net (E1..E256), skip if not found
    try:
        raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    except Exception as e:
        logger.warning(f"Could not set montage: {e}")

    # Bandpass filter
    logger.info(f"Bandpass filtering {config['l_freq']}–{config['h_freq']} Hz")
    raw.filter(config["l_freq"], config["h_freq"], method="fir", verbose=False)

    # Notch filter
    if config.get("notch_freqs"):
        logger.info(f"Notch filtering at {config['notch_freqs']} Hz")
        raw.notch_filter(config["notch_freqs"], verbose=False)

    # Resample
    target = config["target_sfreq"]
    if raw.info["sfreq"] != target:
        logger.info(f"Resampling {raw.info['sfreq']} → {target} Hz")
        raw.resample(target, verbose=False)

    # ICA
    n_comp = config.get("n_ica_components", 20)
    n_comp = min(n_comp, len(raw.ch_names) - 1)
    logger.info(f"Fitting ICA ({n_comp} components)")
    ica = ICA(n_components=n_comp, random_state=42, verbose=False)
    ica.fit(raw, verbose=False)

    try:
        eog_inds, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_inds[:2]
        logger.info(f"ICA: excluding {len(ica.exclude)} EOG component(s)")
    except Exception as e:
        logger.warning(f"Could not auto-detect EOG components: {e}")

    ica.apply(raw, verbose=False)
    logger.info("ICA applied.")
    return raw


def extract_epochs(raw, dataset_card: dict, config: dict, source_path: str | None = None):
    """
    Extract word-level epochs from cleaned raw data.

    For ZuCo 2.0: events.tsv onset is in SAMPLES (not seconds) — divided by sfreq.
    Label comes from the 'type' column (image filename stripped of extension).

    Args:
        raw: Cleaned mne.io.Raw object.
        dataset_card: Parsed dataset card (label location, format, etc.).
        config: Dict matching configs/preprocessing_eeg.yaml schema.
        source_path: Original BrainVision header path. When provided, use this
            instead of `raw.filenames[0]` because MNE may expose the backing
            `.eeg` payload file there for BrainVision recordings.

    Returns:
        Tuple of (mne.Epochs, list[str] labels).

    Raises:
        ValueError: If labels are sentence-level rather than word-level.
    """
    import mne
    import pandas as pd

    vhdr_path = Path(source_path) if source_path is not None else Path(raw.filenames[0])
    events_tsv = vhdr_path.parent / vhdr_path.name.replace("_eeg.vhdr", "_events.tsv")

    if not events_tsv.exists():
        raise FileNotFoundError(f"Events TSV not found: {events_tsv}")

    try:
        events_df = pd.read_csv(str(events_tsv), sep="\t")
    except UnicodeDecodeError:
        logger.warning(f"Falling back to latin1 decoding for events file: {events_tsv}")
        events_df = pd.read_csv(str(events_tsv), sep="\t", encoding="latin1")
    logger.info(f"Events TSV: {len(events_df)} rows, columns={list(events_df.columns)}")

    # ZuCo 2.0: onset column is in data-point samples (not seconds)
    # Determine unit: if max onset >> 1000 assume samples, else seconds
    sfreq = raw.info["sfreq"]
    onset_raw = events_df["onset"].values.astype(float)

    if onset_raw.max() > 1000:
        logger.info(
            f"Interpreting onset as samples (max={onset_raw.max():.0f} >> 1000). "
            f"Converting to seconds by dividing by sfreq={sfreq}."
        )
        onset_s = onset_raw / sfreq
    else:
        logger.info("Interpreting onset as seconds (max < 1000).")
        onset_s = onset_raw

    # Extract labels from 'type' column (image filename without extension)
    label_col = None
    for col in ["type", "trial_type", "stim_file"]:
        if col in events_df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No label column found; available: {list(events_df.columns)}")

    raw_labels = (
        events_df[label_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.\w+$", "", regex=True)  # remove extension
        .str.lower()
        .tolist()
    )

    onset_samples = (onset_s * sfreq).astype(int)
    n_times = len(raw.times)
    valid_mask = (onset_samples >= 0) & (onset_samples < n_times)
    if not valid_mask.all():
        logger.warning(f"Dropping {(~valid_mask).sum()} events outside recording bounds")
    onset_samples = onset_samples[valid_mask]
    raw_labels = [l for l, v in zip(raw_labels, valid_mask) if v]

    if len(onset_samples) == 0:
        raise ValueError("No valid events after filtering by recording bounds")

    events = np.column_stack(
        [
            onset_samples,
            np.zeros(len(onset_samples), dtype=int),
            np.ones(len(onset_samples), dtype=int),
        ]
    )

    baseline = tuple(config["baseline"]) if config.get("baseline") else None
    epochs = mne.Epochs(
        raw,
        events,
        event_id=1,
        tmin=config["epoch_tmin"],
        tmax=config["epoch_tmax"],
        baseline=baseline,
        reject=None,
        preload=True,
        verbose=False,
    )

    labels = [raw_labels[i] for i in epochs.selection]
    logger.info(f"Extracted {len(epochs)} epochs from {len(onset_samples)} events")
    return epochs, labels


def normalize(epochs_data: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each channel across all epochs.

    Mean and std computed across time and epoch dimensions per channel.

    Args:
        epochs_data: float32 array (n_epochs, n_channels, n_timepoints).

    Returns:
        Normalized float32 array, same shape.
    """
    # Compute mean and std across epoch (axis 0) and time (axis 2) per channel
    mean = epochs_data.mean(axis=(0, 2), keepdims=True)
    std = epochs_data.std(axis=(0, 2), keepdims=True)
    return ((epochs_data - mean) / (std + 1e-8)).astype(np.float32)


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
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}_epochs.npz"

    assert data.ndim == 3, f"Expected 3D data, got shape {data.shape}"
    assert len(labels) == data.shape[0], (
        f"Label/epoch count mismatch: {len(labels)} labels vs {data.shape[0]} epochs"
    )
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"

    np.savez(
        str(out_path),
        data=data,
        labels=np.array(labels, dtype=object),
        subject_id=np.array(subject_id),
        sfreq=np.array(sfreq),
        ch_names=np.array(ch_names, dtype=object),
        event_onsets_s=event_onsets_s.astype(np.float64),
    )
    logger.info(f"Saved EEG epochs: {out_path}  shape={data.shape}")
    return str(out_path)


def run_subject(subject_id: str, dataset_name: str, config: dict) -> str:
    """
    Run full EEG preprocessing pipeline for one subject.

    Concatenates all sessions and runs for the subject.

    Args:
        subject_id: Subject identifier (e.g., 'sub-13').
        dataset_name: Dataset name matching data root config.
        config: Parsed preprocessing_eeg.yaml config.

    Returns:
        Path to saved .npz file.
    """
    data_root = Path(config.get("data_root", "data/eeg_zuco"))
    subj_dir = data_root / subject_id

    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

    vhdr_files = sorted(subj_dir.glob("ses-*/eeg/*_eeg.vhdr"))
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr files found under {subj_dir}")

    logger.info(f"Subject {subject_id}: {len(vhdr_files)} VHDR file(s)")

    dataset_card = {}
    all_data = []
    all_labels: list[str] = []
    all_onsets: list[float] = []
    last_ch_names: list[str] = []

    for vhdr in vhdr_files:
        logger.info(f"  Processing: {vhdr.name}")
        try:
            raw = load_raw(str(vhdr), dataset_card)
            raw = preprocess_raw(raw, config)
            epochs, labels = extract_epochs(raw, dataset_card, config, source_path=str(vhdr))

            if len(epochs) == 0:
                logger.warning(f"  No epochs from {vhdr.name} — skipping")
                continue

            data = normalize(epochs.get_data().astype(np.float32))
            all_data.append(data)
            all_labels.extend(labels)
            all_onsets.extend(
                epochs.events[:, 0].astype(float) / raw.info["sfreq"]
            )
            last_ch_names = list(epochs.ch_names)
        except Exception as e:
            logger.error(f"  Failed {vhdr.name}: {e}", exc_info=True)
            continue

    if not all_data:
        raise RuntimeError(f"No data successfully processed for {subject_id}")

    combined_data = np.concatenate(all_data, axis=0)
    event_onsets = np.array(all_onsets, dtype=np.float64)
    output_dir = str(Path(config["output_dir"]) / dataset_name / "eeg")

    return save_epochs(
        combined_data,
        all_labels,
        subject_id,
        config["target_sfreq"],
        last_ch_names,
        event_onsets,
        output_dir,
    )


def run_all_subjects(dataset_name: str, config_path: str) -> list[str]:
    """
    Run EEG preprocessing for all subjects specified in config.

    Args:
        dataset_name: Dataset name.
        config_path: Path to configs/preprocessing_eeg.yaml.

    Returns:
        List of paths to saved .npz files.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = Path(config.get("data_root", "data/eeg_zuco"))
    subjects = config.get("subjects", "all")

    if subjects == "all":
        subjects = sorted(
            [d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        )

    logger.info(f"Running EEG preprocessing for {len(subjects)} subject(s): {subjects}")
    paths = []
    for subj in subjects:
        try:
            path = run_subject(subj, dataset_name, config)
            paths.append(path)
            logger.info(f"✓ {subj} → {path}")
        except Exception as e:
            logger.error(f"✗ {subj}: {e}")
    return paths


if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "workers/worker_data/configs/preprocessing_eeg.yaml"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "zuco"
    run_all_subjects(dataset_name, config_path)
