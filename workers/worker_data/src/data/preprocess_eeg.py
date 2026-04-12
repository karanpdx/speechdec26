"""
EEG preprocessing pipeline.

Transforms raw EEG recordings into clean, epoched, normalized tensors.

Supported formats:
- BrainVision (.vhdr): ZuCo-style BIDS layout with matching *_events.tsv
- MATLAB (.mat):       FieldTrip-style struct with data.eeg / data.event / data.fsample

Output schema (saved as .npz):
    data:           float32  (n_epochs, n_channels, n_timepoints)
    labels:         list[str]  length n_epochs
    subject_id:     str
    sfreq:          float
    ch_names:       list[str]
    event_onsets_s: float64  (n_epochs,)
"""

import logging
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _extract_audio_mat_channel_names(data_struct) -> list[str]:
    raw_names = np.asarray(data_struct.dim[0, 0].chan[0, 0].eeg[0, 0]).ravel()
    ch_names = []
    for item in raw_names:
        if isinstance(item, np.ndarray):
            ch_names.append(str(item.ravel()[0]))
        else:
            ch_names.append(str(item))
    return ch_names


def _load_audio_preproc_mat(file_path: str, drop_aux_channels: bool = True) -> tuple[np.ndarray, float, list[str], np.ndarray]:
    """
    Load the local `data-eeg` MATLAB derivative format.

    The files are already preprocessed and epoched:
      - 60 trials per subject
      - each trial stored as (time, channels)
      - fsample.eeg = 64
      - channel labels include 64 EEG channels plus EXG1/EXG2

    Returns:
        data: float32 (n_trials, n_channels, n_timepoints)
        sfreq: float
        ch_names: list[str]
        event_codes: int array (n_trials,)
    """
    from scipy.io import loadmat

    mat = loadmat(str(file_path), squeeze_me=False, struct_as_record=False)
    data_struct = mat["data"][0, 0]

    sfreq = float(np.asarray(data_struct.fsample[0, 0].eeg).ravel()[0])
    ch_names = _extract_audio_mat_channel_names(data_struct)

    trials = []
    for trial in np.asarray(data_struct.eeg).ravel():
        arr = np.asarray(trial, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D trial array, got shape {arr.shape}")
        trials.append(arr.T)  # stored as (time, channels) → pipeline expects (channels, time)

    data = np.stack(trials, axis=0).astype(np.float32)
    event_codes = np.array(
        [int(ev.value[0, 0][0, 0]) for ev in np.asarray(data_struct.event[0, 0].eeg).ravel()],
        dtype=np.int64,
    )

    if drop_aux_channels:
        keep_indices = [i for i, name in enumerate(ch_names) if not name.upper().startswith("EXG")]
        data = data[:, keep_indices, :]
        ch_names = [ch_names[i] for i in keep_indices]

    logger.info(
        "Loaded audio-preproc EEG MAT %s with shape=%s sfreq=%.1f channels=%d",
        file_path,
        tuple(data.shape),
        sfreq,
        len(ch_names),
    )
    return data, sfreq, ch_names, event_codes


def _load_trial_labels_from_file(label_path: str, subject_id: str, n_trials: int) -> list[str]:
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"Trial label file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if subject_id in payload:
                labels = payload[subject_id]
            elif "labels" in payload:
                labels = payload["labels"]
            else:
                raise ValueError(f"JSON label file {path} does not contain subject '{subject_id}' or top-level 'labels'")
        else:
            labels = payload
    else:
        import pandas as pd

        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if "subject_id" in df.columns:
            df = df[df["subject_id"].astype(str) == str(subject_id)]
        label_col = next((col for col in ("label", "word", "trial_type") if col in df.columns), None)
        if label_col is None:
            raise ValueError(f"Label file {path} must contain one of: label, word, trial_type")
        labels = df[label_col].astype(str).tolist()

    labels = [str(label).strip().lower() for label in labels]
    if len(labels) != n_trials:
        raise ValueError(f"Expected {n_trials} trial labels for {subject_id}, got {len(labels)} from {path}")
    return labels


def _resolve_audio_preproc_labels(event_codes: np.ndarray, config: dict, subject_id: str) -> list[str]:
    """
    Resolve word labels for the MATLAB derivative dataset.

    The local files only expose binary event codes by default, so true word labels must
    come from an external trial-label file or an explicit event-code mapping.
    """
    n_trials = int(len(event_codes))

    label_path = config.get("trial_label_path")
    if label_path:
        return _load_trial_labels_from_file(label_path, subject_id, n_trials)

    event_label_map = config.get("event_label_map")
    if event_label_map:
        labels = []
        for code in event_codes.tolist():
            label = event_label_map.get(code, event_label_map.get(str(code)))
            if label is None:
                raise ValueError(f"No label mapping found for event code {code}")
            labels.append(str(label).strip().lower())
        return labels

    if config.get("require_word_labels", True):
        raise ValueError(
            "This EEG MATLAB derivative only exposes binary event codes (1/2), not word labels. "
            "Provide `trial_label_path` with per-trial word labels or disable `require_word_labels` "
            "to emit condition labels for debugging only."
        )

    return [f"condition_{code}" for code in event_codes.tolist()]


def load_raw(file_path: str, dataset_card: dict):
    """
    Load a raw EEG file. Format is detected from the file extension.

    Supports:
    - .vhdr: BrainVision (ZuCo BIDS layout)
    - .mat:  FieldTrip MATLAB struct (BioSemi ActiveTwo)

    For .mat files, trigger event times (in seconds) and trigger code values are
    stored in ``dataset_card`` under ``_mat_event_times_s`` and ``_mat_event_values``
    so they survive any in-place raw transformations before epoch extraction.

    Args:
        file_path: Path to the raw EEG file.
        dataset_card: Mutable dict; .mat loader writes event metadata here.

    Returns:
        mne.io.Raw object with data loaded into memory.
    """
    import mne

    path = Path(file_path)

    if path.suffix == ".vhdr":
        logger.info(f"Loading BrainVision EEG: {file_path}")
        raw = mne.io.read_raw_brainvision(str(file_path), preload=False, verbose=False)
        return raw

    if path.suffix == ".mat":
        return _load_raw_mat(path, dataset_card)

    raise ValueError(f"Unsupported EEG format {path.suffix!r} — expected .vhdr or .mat")


def _load_raw_mat(path: Path, dataset_card: dict):
    """
    Load a FieldTrip-style .mat EEG file into an MNE RawArray.

    Expected MATLAB struct layout::

        data.eeg             float64 (n_times, n_channels)  — data in µV
        data.fsample.eeg     scalar  — sampling rate in Hz
        data.dim.chan.eeg    str[]   — channel names
        data.event.eeg.sample  int[] — event onset samples (at original sfreq)
        data.event.eeg.value   int[] — trigger code per event

    EXG* channels are typed as 'eog'; Status is typed as 'stim'; all others
    are typed as 'eeg'.  EEG channel values are scaled from µV to V for MNE.
    """
    import mne
    import scipy.io as sio

    logger.info(f"Loading MATLAB EEG: {path}")
    mat = sio.loadmat(str(path), simplify_cells=True)
    ds = mat["data"]

    eeg_arr = ds["eeg"].astype(np.float64)  # (n_times, n_channels), µV
    sfreq = float(ds["fsample"]["eeg"])
    ch_names = list(ds["dim"]["chan"]["eeg"])

    ch_types = []
    for ch in ch_names:
        if ch.startswith("EXG"):
            ch_types.append("eog")
        elif ch == "Status":
            ch_types.append("stim")
        else:
            ch_types.append("eeg")

    # Scale EEG channels µV → V (MNE internal convention)
    eeg_idx = [i for i, t in enumerate(ch_types) if t == "eeg"]
    eeg_arr[:, eeg_idx] *= 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_arr.T, info, verbose=False)

    # Store events as seconds (rate-independent) so they survive resampling
    ev = ds["event"]["eeg"]
    ev_samples = np.array(ev["sample"], dtype=int)
    ev_values = np.array(ev["value"], dtype=int)
    dataset_card["_mat_event_times_s"] = ev_samples / sfreq
    dataset_card["_mat_event_values"] = ev_values

    logger.info(
        f"  {path.name}: {len(eeg_idx)} EEG channels, sfreq={sfreq} Hz, "
        f"duration={raw.times[-1]:.1f}s, {len(ev_samples)} events"
    )
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
    Extract stimulus-locked epochs from cleaned raw data.

    Routes to the appropriate extractor based on data origin:

    * **.mat files** — uses trigger codes cached in *dataset_card* by
      :func:`_load_raw_mat`.  Stimulus codes and an optional code→label
      mapping are read from *config*.

    * **.vhdr files** — reads the matching ``*_events.tsv`` sidecar.
      Onset column is assumed to be in SAMPLES when its max value > 1000
      (ZuCo 2.0 convention).  Label comes from the first available column
      among ``type``, ``trial_type``, ``stim_file``.

    Args:
        raw: Cleaned mne.io.Raw object.
        dataset_card: Dict from :func:`load_raw`; contains ``_mat_event_*``
            keys for .mat files.
        config: Dict matching configs/preprocessing_eeg.yaml schema.
        source_path: Original .vhdr header path (ignored for .mat files).

    Returns:
        Tuple of (mne.Epochs, list[str] labels).

    Raises:
        FileNotFoundError: If the events TSV is missing for a .vhdr file.
        ValueError: If no valid events remain after filtering.
    """
    if "_mat_event_times_s" in dataset_card:
        return _extract_epochs_mat(raw, dataset_card, config)
    return _extract_epochs_vhdr(raw, config, source_path)


def _extract_epochs_mat(raw, dataset_card: dict, config: dict):
    """Extract epochs from a .mat raw using cached trigger code metadata."""
    import mne

    sfreq = raw.info["sfreq"]  # may be resampled
    event_times_s = dataset_card["_mat_event_times_s"].copy()
    event_values = dataset_card["_mat_event_values"].copy()

    # Filter to configured stimulus codes (None → use all events)
    stimulus_codes = config.get("stimulus_codes")
    if stimulus_codes is not None:
        mask = np.isin(event_values, stimulus_codes)
        event_times_s = event_times_s[mask]
        event_values = event_values[mask]
        logger.info(
            f"Filtered to stimulus_codes={stimulus_codes}: "
            f"{mask.sum()}/{len(mask)} events kept"
        )

    if len(event_times_s) == 0:
        raise ValueError(
            "No events remain after stimulus_code filtering. "
            "Check 'stimulus_codes' in the preprocessing config."
        )

    # Convert seconds → samples at (possibly resampled) rate
    onset_samples = (event_times_s * sfreq).astype(int)
    n_times = len(raw.times)
    valid = (onset_samples >= 0) & (onset_samples < n_times)
    if not valid.all():
        logger.warning(f"Dropping {(~valid).sum()} events outside recording bounds")
    onset_samples = onset_samples[valid]
    event_values = event_values[valid]

    if len(onset_samples) == 0:
        raise ValueError("No valid events within recording bounds after resampling")

    # Map trigger code → label string (fallback: "code_<N>")
    code_to_label: dict = config.get("event_code_labels") or {}
    raw_labels = [
        str(code_to_label.get(int(v), f"code_{v}")) for v in event_values
    ]

    events = np.column_stack([
        onset_samples,
        np.zeros(len(onset_samples), dtype=int),
        np.ones(len(onset_samples), dtype=int),
    ])

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


def _extract_epochs_vhdr(raw, config: dict, source_path: str | None):
    """
    Extract word-level epochs from a BrainVision raw using *_events.tsv.

    For ZuCo 2.0: events.tsv onset is in SAMPLES (not seconds) — divided by sfreq.
    Label comes from the 'type' column (image filename stripped of extension).
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

    Handles two directory layouts automatically:

    * **Flat .mat layout** (e.g. ``EEG/S1.mat``): ``data_root/<subject_id>.mat``
    * **BIDS .vhdr layout** (e.g. ZuCo): ``data_root/<subject_id>/ses-*/eeg/*_eeg.vhdr``

    Args:
        subject_id: Subject identifier (e.g. 'S1' for .mat, 'sub-13' for BIDS).
        dataset_name: Dataset name matching data root config.
        config: Parsed preprocessing_eeg.yaml config.

    Returns:
        Path to saved .npz file.
    """
    data_root = Path(config.get("data_root", "data/eeg_zuco"))
    dataset_style = config.get("dataset_style", "auto")

    if dataset_style == "audio_preproc_mat" or (
        dataset_style == "auto" and any(data_root.glob("S*_data_preproc.mat"))
    ):
        mat_path = data_root / f"{subject_id}_data_preproc.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        data, sfreq, ch_names, event_codes = _load_audio_preproc_mat(
            str(mat_path),
            drop_aux_channels=bool(config.get("drop_aux_channels", True)),
        )
        labels = _resolve_audio_preproc_labels(event_codes, config, subject_id)
        data = normalize(data)
        event_onsets = np.zeros(data.shape[0], dtype=np.float64)
        output_dir = str(Path(config["output_dir"]) / dataset_name / "eeg")
        return save_epochs(
            data,
            labels,
            subject_id,
            sfreq,
            ch_names,
            event_onsets,
            output_dir,
        )

    # Auto-detect format: .mat flat layout takes priority
    mat_file = data_root / f"{subject_id}.mat"
    if mat_file.exists():
        eeg_files = [mat_file]
        logger.info(f"Subject {subject_id}: using .mat file {mat_file}")
    else:
        subj_dir = data_root / subject_id
        if not subj_dir.exists():
            raise FileNotFoundError(
                f"Neither {mat_file} nor subject directory {subj_dir} found"
            )
        eeg_files = sorted(subj_dir.glob("ses-*/eeg/*_eeg.vhdr"))
        if not eeg_files:
            raise FileNotFoundError(f"No .vhdr files found under {subj_dir}")
        logger.info(f"Subject {subject_id}: {len(eeg_files)} VHDR file(s)")

    all_data = []
    all_labels: list[str] = []
    all_onsets: list[float] = []
    last_ch_names: list[str] = []

    for eeg_file in eeg_files:
        logger.info(f"  Processing: {eeg_file.name}")
        dataset_card: dict = {}
        try:
            raw = load_raw(str(eeg_file), dataset_card)
            raw = preprocess_raw(raw, config)
            epochs, labels = extract_epochs(
                raw, dataset_card, config, source_path=str(eeg_file)
            )

            if len(epochs) == 0:
                logger.warning(f"  No epochs from {eeg_file.name} — skipping")
                continue

            data = normalize(epochs.get_data().astype(np.float32))
            all_data.append(data)
            all_labels.extend(labels)
            all_onsets.extend(
                epochs.events[:, 0].astype(float) / raw.info["sfreq"]
            )
            last_ch_names = list(epochs.ch_names)
        except Exception as e:
            logger.error(f"  Failed {eeg_file.name}: {e}", exc_info=True)
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

    Subject discovery:
    - If ``subjects: "all"`` and .mat files exist in ``data_root``, subjects are
      inferred from the .mat filenames (e.g. ``S1.mat`` → ``S1``).
    - Otherwise, ``sub-*`` subdirectories are used (BIDS layout).

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
    dataset_style = config.get("dataset_style", "auto")
    subjects = config.get("subjects", "all")

    if subjects == "all" and (
        dataset_style == "audio_preproc_mat" or (dataset_style == "auto" and any(data_root.glob("S*_data_preproc.mat")))
    ):
        subjects = sorted(p.name.replace("_data_preproc.mat", "") for p in data_root.glob("S*_data_preproc.mat"))
    elif subjects == "all":
        mat_files = sorted(data_root.glob("*.mat"))
        if mat_files:
            subjects = [f.stem for f in mat_files]
            logger.info(f"Auto-discovered {len(subjects)} .mat subjects: {subjects}")
        else:
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
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "eeg_mat"
    run_all_subjects(dataset_name, config_path)
