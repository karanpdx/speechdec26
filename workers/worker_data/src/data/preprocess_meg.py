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

import ast
import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _kit_headshape_path(path: Path | None) -> str | None:
    """Return a KIT-compatible headshape path, mirroring .pos files to .txt."""
    if path is None:
        return None
    if path.suffix != ".pos":
        return str(path)

    with path.open(encoding="utf-8") as src:
        contents = src.read()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix=path.stem + "_",
        delete=False,
    ) as handle:
        handle.write(contents)
        return handle.name


def load_raw(file_path: str, dataset_card: dict):
    """
    Load a raw MEG file.

    Args:
        file_path: Path to the raw MEG file.
        dataset_card: Parsed DATASET_CARD.md as a dict.

    Returns:
        mne.io.Raw object.
    """
    import mne

    p = Path(file_path)
    # Find matching .mrk marker file (same dir, same prefix)
    mrk_path = p.parent / p.name.replace("_meg.con", "_markers.mrk")
    mrk = str(mrk_path) if mrk_path.exists() else None
    elp_path = next(iter(sorted(p.parent.glob("*_acq-ELP_headshape.pos"))), None)
    hsp_path = next(iter(sorted(p.parent.glob("*_acq-HSP_headshape.pos"))), None)
    elp = _kit_headshape_path(elp_path)
    hsp = _kit_headshape_path(hsp_path)

    logger.info(f"Loading KIT MEG file: {file_path} (mrk={mrk}, elp={elp}, hsp={hsp})")
    raw = mne.io.read_raw_kit(
        str(file_path),
        mrk=mrk,
        elp=elp,
        hsp=hsp,
        preload=False,
        verbose=False,
    )
    return raw


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
    # MASC-MEG uses KIT/Yokogawa system — Maxwell filtering is Elekta-only
    logger.warning(
        "Maxwell filtering is not applicable to KIT/Yokogawa systems. "
        "Skipping SSS. Set apply_maxwell: false in config to suppress this warning."
    )
    return raw


def annotate_movement_artifacts(raw):
    """
    Annotate segments with excessive head movement (if head position data available).

    Args:
        raw: mne.io.Raw MEG object.

    Returns:
        Raw with movement artifact annotations added (or unchanged if no
        head position data available — logs a warning).
    """
    # KIT system does not record continuous head position (no cHPI)
    logger.warning(
        "No continuous HPI data available for KIT system — skipping movement annotation."
    )
    return raw


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
    from mne.preprocessing import ICA

    raw.load_data(verbose=False)
    raw.pick_types(meg=True, eeg=False, stim=False, exclude="bads")
    logger.info(f"Channels after pick: {len(raw.ch_names)}, sfreq={raw.info['sfreq']}")

    # Maxwell filter (skipped for KIT)
    if config.get("apply_maxwell", False):
        raw = apply_maxwell_filter(
            raw,
            crosstalk_file=config.get("crosstalk_file"),
            fine_cal_file=config.get("fine_cal_file"),
        )

    # Movement artifact annotation (skipped for KIT)
    raw = annotate_movement_artifacts(raw)

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
    logger.info(f"Fitting ICA with {n_comp} components")
    ica = ICA(n_components=n_comp, random_state=42, verbose=False)
    ica.fit(raw, verbose=False)

    try:
        eog_inds, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_inds[:2]
        logger.info(f"ICA: excluding {len(ica.exclude)} EOG component(s): {ica.exclude}")
    except Exception as e:
        logger.warning(f"Could not auto-detect EOG components: {e}")

    ica.apply(raw, verbose=False)
    logger.info("ICA applied.")
    return raw


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
    import mne
    import pandas as pd

    con_path = Path(raw.filenames[0])
    events_tsv = con_path.parent / con_path.name.replace("_meg.con", "_events.tsv")
    if not events_tsv.exists():
        raise FileNotFoundError(f"Events TSV not found: {events_tsv}")

    events_df = pd.read_csv(str(events_tsv), sep="\t")

    # trial_type column contains Python-dict strings with 'kind' and 'word' fields
    def parse_trial_type(s):
        try:
            return ast.literal_eval(str(s))
        except Exception:
            return {}

    events_df["parsed"] = events_df["trial_type"].apply(parse_trial_type)
    word_rows = events_df[
        events_df["parsed"].apply(lambda x: x.get("kind") == "word")
    ].copy()

    if len(word_rows) == 0:
        raise ValueError(
            f"No word-level events found in {events_tsv}. "
            "Check that trial_type column contains dicts with 'kind'='word'."
        )

    logger.info(f"Found {len(word_rows)} word-level events")

    sfreq = raw.info["sfreq"]
    onset_s = word_rows["onset"].values.astype(float)
    onset_samples = (onset_s * sfreq).astype(int)

    # Clip to recording bounds
    n_times = len(raw.times)
    valid_mask = (onset_samples >= 0) & (onset_samples < n_times)
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        logger.warning(f"Dropping {n_dropped} events outside recording bounds")
    onset_samples = onset_samples[valid_mask]
    word_rows = word_rows[valid_mask]

    labels_all = [row.get("word", "") for row in word_rows["parsed"].values]

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

    # Use epochs.selection (indices into events/labels_all) to align labels
    labels = [labels_all[i] for i in epochs.selection]
    logger.info(f"Extracted {len(epochs)} epochs ({len(epochs.ch_names)} channels)")
    return epochs, labels


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
    logger.info(f"Saved MEG epochs: {out_path}  shape={data.shape}")
    return str(out_path)


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
    data_root = Path(config.get("data_root", "ag3kj/osfstorage"))
    subj_dir = data_root / subject_id

    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

    con_files = sorted(subj_dir.glob("ses-*/meg/*_meg.con"))
    if not con_files:
        raise FileNotFoundError(f"No .con files found under {subj_dir}")

    logger.info(f"Subject {subject_id}: found {len(con_files)} .con file(s)")

    dataset_card = {}
    all_data = []
    all_labels: list[str] = []
    all_onsets: list[float] = []
    last_ch_names: list[str] = []

    for con_file in con_files:
        logger.info(f"  Processing: {con_file.name}")
        try:
            raw = load_raw(str(con_file), dataset_card)
            raw = preprocess_raw(raw, config)
            epochs, labels = extract_epochs(raw, dataset_card, config)

            if len(epochs) == 0:
                logger.warning(f"  No epochs extracted from {con_file.name} — skipping")
                continue

            data = normalize(epochs.get_data().astype(np.float32))
            all_data.append(data)
            all_labels.extend(labels)
            # Convert epoch event samples back to seconds
            all_onsets.extend(epochs.events[:, 0].astype(float) / raw.info["sfreq"])
            last_ch_names = list(epochs.ch_names)
        except Exception as e:
            logger.error(f"  Failed {con_file.name}: {e}", exc_info=True)
            continue

    if not all_data:
        raise RuntimeError(f"No data successfully processed for {subject_id}")

    combined_data = np.concatenate(all_data, axis=0)
    event_onsets = np.array(all_onsets, dtype=np.float64)
    output_dir = str(Path(config["output_dir"]) / dataset_name / "meg")

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
    Run MEG preprocessing for all subjects specified in config.

    Args:
        dataset_name: Dataset name.
        config_path: Path to configs/preprocessing_meg.yaml.

    Returns:
        List of paths to saved .npz files.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = Path(config.get("data_root", "ag3kj/osfstorage"))
    subjects = config.get("subjects", "all")

    if subjects == "all":
        subjects = sorted(
            [d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        )

    logger.info(f"Running MEG preprocessing for {len(subjects)} subject(s): {subjects}")
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "workers/worker_data/configs/preprocessing_meg.yaml"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "masc_meg"
    run_all_subjects(dataset_name, config_path)
