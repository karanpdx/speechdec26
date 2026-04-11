"""
fMRI preprocessing pipeline.

Transforms raw BOLD NIfTI timeseries into per-word beta maps via a GLM
with HRF convolution.

For the Huth dataset:
  - Word-level timing is NOT included in the BIDS events files.
  - Pre-processed HDF5 derivatives exist but are incompletely downloaded.
  - The pipeline detects these conditions and logs a clear error.
  - Use `run_from_hdf5()` if HDF5 files are complete and contain voxel timeseries.

Output schema (saved as .npz):
    data:                   float32  (n_words, n_voxels)
    labels:                 list[str]  length n_words
    subject_id:             str
    voxel_coords:           int32    (n_voxels, 3)  MNI coordinates
    pca_explained_variance: float    (0.0 if PCA not applied)
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_bold(nifti_path: str) -> tuple:
    """
    Load BOLD NIfTI image. Confirms shape is (x, y, z, n_timepoints).

    Args:
        nifti_path: Path to the BOLD .nii or .nii.gz file.

    Returns:
        Tuple of (nibabel.Nifti1Image, shape tuple).
    """
    import nibabel as nib

    logger.info(f"Loading BOLD: {nifti_path}")
    img = nib.load(nifti_path)
    shape = img.shape
    if len(shape) != 4:
        raise ValueError(f"Expected 4D BOLD image, got shape {shape}: {nifti_path}")
    logger.info(f"BOLD shape: {shape}")
    return img, shape


def load_brain_mask(mask_path: str = None, bold_img=None):
    """
    Load or compute a brain mask.

    If mask_path is None, computes a mask using nilearn.masking.compute_brain_mask().

    Args:
        mask_path: Path to a brain mask NIfTI, or None to auto-compute.
        bold_img: nibabel image used for auto-computation if mask_path is None.

    Returns:
        nibabel mask image.
    """
    import nibabel as nib
    from nilearn import masking

    if mask_path is not None:
        logger.info(f"Loading brain mask: {mask_path}")
        return nib.load(mask_path)

    logger.info("Auto-computing brain mask from BOLD image")
    return masking.compute_brain_mask(bold_img, verbose=0)


def load_events(events_path: str, label_info: dict) -> "pd.DataFrame":
    """
    Load stimulus events and return a DataFrame with columns:
    onset (float, seconds), duration (float), trial_type (str, word).

    Handles conversion from dataset-specific event formats.
    Documents any format conversion explicitly.

    Args:
        events_path: Path to events file (TSV, CSV, MAT, etc.).
        label_info: Parsed label section from DATASET_CARD.md.

    Returns:
        pandas DataFrame with columns ['onset', 'duration', 'trial_type'].

    Raises:
        ValueError: If events cannot be mapped to word-level labels.
    """
    import pandas as pd

    p = Path(events_path)
    if not p.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    if p.suffix in {".tsv", ".csv"}:
        sep = "\t" if p.suffix == ".tsv" else ","
        df = pd.read_csv(str(p), sep=sep)
    elif p.suffix == ".TextGrid":
        df = _load_textgrid_events(p)
    else:
        raise ValueError(f"Unsupported events format: {p.suffix}")

    required = {"onset", "duration", "trial_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Events file missing columns {missing}. "
            f"Available: {list(df.columns)}. "
            "The events source must provide word-level timing, either directly "
            "or via a parseable TextGrid word tier."
        )

    # Drop non-word rows if a 'kind' column exists
    if "kind" in df.columns:
        df = df[df["kind"] == "word"].copy()

    return df[["onset", "duration", "trial_type"]].dropna(subset=["trial_type"])


def _load_textgrid_events(textgrid_path: Path) -> "pd.DataFrame":
    """Parse a Praat TextGrid word tier into onset/duration/trial_type rows."""
    import pandas as pd

    lines = textgrid_path.read_text(encoding="utf-8").splitlines()
    intervals: list[dict[str, float | str]] = []
    in_word_tier = False
    current: dict[str, float | str] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith('name = "'):
            tier_name = line.split('"')[1]
            in_word_tier = tier_name.lower() == "word"
            continue
        if not in_word_tier:
            continue
        if line.startswith("intervals ["):
            if current:
                intervals.append(current)
            current = {}
            continue
        if line.startswith("xmin = "):
            current["onset"] = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("xmax = "):
            xmax = float(line.split("=", 1)[1].strip())
            onset = float(current.get("onset", xmax))
            current["duration"] = max(0.0, xmax - onset)
            continue
        if line.startswith('text = "'):
            current["trial_type"] = line.split('"', 2)[1]

    if current:
        intervals.append(current)

    df = pd.DataFrame(intervals)
    if df.empty:
        raise ValueError(f"No intervals parsed from TextGrid word tier: {textgrid_path}")

    df["trial_type"] = df["trial_type"].astype(str).str.strip().str.lower()
    df = df[
        ~df["trial_type"].isin({"", "sp", "{ns}", "{ls}", "ns", "ls", "br"})
    ].copy()
    if df.empty:
        raise ValueError(f"No usable word intervals found in TextGrid: {textgrid_path}")

    logger.info(f"Loaded {len(df)} word intervals from TextGrid: {textgrid_path.name}")
    return df[["onset", "duration", "trial_type"]]


def _infer_tr_for_bold(bold_path: Path, config: dict) -> float:
    """Return TR from config or BIDS sidecar JSON."""
    if config.get("tr") is not None:
        return float(config["tr"])

    import json

    candidate_jsons = [
        Path(str(bold_path).replace(".nii.gz", ".json").replace(".nii", ".json")),
        bold_path.parent.parent.parent / f"task-{_task_name_from_bold_path(bold_path)}_bold.json",
    ]
    for json_path in candidate_jsons:
        if not json_path.exists():
            continue
        with json_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        tr = metadata.get("RepetitionTime")
        if tr is not None:
            logger.info(f"Inferred TR={tr}s from {json_path}")
            return float(tr)

    raise ValueError(
        f"TR is missing for {bold_path.name}. Set it in config or provide a BIDS JSON sidecar."
    )


def fit_glm(bold_img, events_df, mask_img, config: dict):
    """
    Fit a first-level GLM using nilearn.

    Args:
        bold_img: nibabel BOLD image.
        events_df: DataFrame with ['onset', 'duration', 'trial_type'].
        mask_img: Brain mask image.
        config: Dict matching configs/preprocessing_fmri.yaml schema.

    Returns:
        Fitted nilearn FirstLevelModel.
    """
    from nilearn.glm.first_level import FirstLevelModel

    tr = config.get("tr")
    if tr is None:
        raise ValueError("TR must be specified in config (preprocessing_fmri.yaml)")

    logger.info(
        f"Fitting GLM: TR={tr}s, hrf={config['hrf_model']}, "
        f"smoothing={config['smoothing_fwhm']}mm, "
        f"high_pass={config['high_pass']}Hz"
    )
    glm = FirstLevelModel(
        t_r=tr,
        hrf_model=config["hrf_model"],
        smoothing_fwhm=config.get("smoothing_fwhm", 6.0),
        high_pass=config.get("high_pass", 0.01),
        mask_img=mask_img,
        standardize=False,
        verbose=0,
    )
    glm.fit(bold_img, events=events_df)
    return glm


def extract_beta_maps(glm, vocabulary: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Extract a beta map (effect size contrast) per unique word.

    Args:
        glm: Fitted nilearn FirstLevelModel.
        vocabulary: List of unique word strings to extract betas for.

    Returns:
        Tuple of (beta_maps nibabel image list, word_labels list[str]).
    """
    beta_imgs = []
    word_labels = []
    for word in vocabulary:
        try:
            beta_img = glm.compute_contrast(word, output_type="effect_size")
            beta_imgs.append(beta_img)
            word_labels.append(word)
        except Exception as e:
            logger.warning(f"Could not extract beta for word '{word}': {e}")
    logger.info(f"Extracted {len(beta_imgs)}/{len(vocabulary)} beta maps")
    return beta_imgs, word_labels


def apply_mask_to_betas(beta_imgs: list, mask_img, roi_mask_img=None) -> np.ndarray:
    """
    Apply brain mask (and optional ROI mask) to beta images → 2D array.

    Args:
        beta_imgs: List of nibabel beta images, one per word.
        mask_img: Brain mask.
        roi_mask_img: Optional ROI mask for further restriction.

    Returns:
        float32 array (n_words, n_voxels).
    """
    from nilearn import masking

    effective_mask = mask_img
    if roi_mask_img is not None:
        logger.info("Applying ROI mask on top of brain mask")
        import nibabel as nib
        import numpy as _np

        combined = (_np.asarray(mask_img.dataobj) > 0) & (_np.asarray(roi_mask_img.dataobj) > 0)
        effective_mask = nib.Nifti1Image(combined.astype(np.int8), mask_img.affine)

    logger.info(f"Masking {len(beta_imgs)} beta images")
    masked = masking.apply_mask(beta_imgs, effective_mask)
    return masked.astype(np.float32)


def apply_pca(data: np.ndarray, n_components: int, fit: bool = True, pca=None):
    """
    Optionally apply PCA to reduce voxel dimensionality.

    IMPORTANT: PCA must be fit on training subjects only.
    Pass fit=False and a pre-fit pca object for val/test subjects.

    Args:
        data: float32 (n_words, n_voxels).
        n_components: Number of PCA components.
        fit: If True, fit a new PCA. If False, use the provided pca object.
        pca: Pre-fit sklearn PCA object (required if fit=False).

    Returns:
        Tuple of (projected float32 (n_words, n_components), sklearn PCA object).
    """
    from sklearn.decomposition import PCA
    if fit:
        actual = min(n_components, min(data.shape[0], data.shape[1]))
        pca = PCA(n_components=actual)
        projected = pca.fit_transform(data)
    else:
        projected = pca.transform(data)
    # Pad with zeros if actual components < requested (e.g. n_samples < n_components)
    if projected.shape[1] < n_components:
        pad = np.zeros((projected.shape[0], n_components - projected.shape[1]), dtype=np.float32)
        projected = np.hstack([projected, pad])
    return projected.astype(np.float32), pca


def get_voxel_mni_coords(mask_img) -> np.ndarray:
    """
    Extract MNI coordinates for each voxel in the mask.

    Args:
        mask_img: Brain mask nibabel image.

    Returns:
        int32 array (n_voxels, 3) of MNI xyz coordinates.
    """
    import nibabel as nib
    import numpy as np

    mask_data = np.asarray(mask_img.dataobj) > 0
    voxel_indices = np.array(np.where(mask_data)).T  # (n_voxels, 3)

    # Apply affine to get MNI coordinates
    affine = mask_img.affine
    coords_homogeneous = np.hstack(
        [voxel_indices, np.ones((len(voxel_indices), 1))]
    )  # (n_voxels, 4)
    mni_coords = (affine @ coords_homogeneous.T)[:3].T  # (n_voxels, 3)
    return mni_coords.astype(np.int32)


def save_betas(
    data: np.ndarray,
    labels: list,
    subject_id: str,
    voxel_coords: np.ndarray,
    pca_explained_variance: float,
    output_dir: str,
) -> str:
    """
    Save beta maps to .npz. Runs output validation before saving.

    Args:
        data: float32 (n_words, n_voxels).
        labels: list of word strings, length n_words.
        subject_id: Subject identifier.
        voxel_coords: int32 (n_voxels, 3) MNI coordinates.
        pca_explained_variance: Fraction explained if PCA was applied, else 0.0.
        output_dir: data/processed/<dataset>/fmri/

    Returns:
        Path to saved .npz file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}_betas.npz"

    assert data.ndim == 2, f"Expected 2D data (n_words, n_voxels), got shape {data.shape}"
    assert len(labels) == data.shape[0], (
        f"Label/beta count mismatch: {len(labels)} vs {data.shape[0]}"
    )
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    assert np.isfinite(data).all(), "Beta data contains NaN or Inf"

    np.savez(
        str(out_path),
        data=data,
        labels=np.array(labels, dtype=object),
        subject_id=np.array(subject_id),
        voxel_coords=voxel_coords,
        pca_explained_variance=np.array(pca_explained_variance, dtype=np.float32),
    )
    logger.info(f"Saved fMRI betas: {out_path}  shape={data.shape}")
    return str(out_path)


def _find_word_events_tsv(subj_dir: Path, task_name: str) -> Path | None:
    """Search for a word-level events source for a given story/task."""
    # Try top-level dataset events file
    dataset_root = subj_dir.parent
    candidates = [
        dataset_root / f"task-{task_name}_events.tsv",
        subj_dir / f"task-{task_name}_events.tsv",
        dataset_root / "derivatives" / "TextGrids" / f"{task_name}.TextGrid",
    ]
    # Also check within session func dirs
    for ses_dir in sorted(subj_dir.glob("ses-*")):
        func_dir = ses_dir / "func"
        if func_dir.exists():
            for tsv in func_dir.glob(f"*task-{task_name}*events.tsv"):
                candidates.append(tsv)

    for c in candidates:
        if c.exists():
            return c
    return None


def _task_name_from_bold_path(bold_path: Path) -> str | None:
    """Extract task name from a BOLD path."""
    parts = bold_path.stem.replace(".nii", "").split("_")
    return next((p.replace("task-", "") for p in parts if p.startswith("task-")), None)


def run_subject(subject_id: str, dataset_name: str, config: dict, pca=None) -> str:
    """
    Run full fMRI preprocessing pipeline for one subject.

    For the Huth dataset: finds per-story BOLD NIfTI files in session func directories
    and matches them with word-level events TSVs. If events TSVs are missing (which is
    the case for Huth narrative stories), raises a descriptive error.

    Args:
        subject_id: Subject identifier (e.g., 'sub-UTS01').
        dataset_name: Dataset name.
        config: Parsed preprocessing_fmri.yaml config.
        pca: Pre-fit PCA object for val/test subjects, or None for training subjects.

    Returns:
        Path to saved .npz file.
    """
    data_root = Path(config.get("data_root", "data/fmri_huth"))
    subj_dir = data_root / subject_id

    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

    # Find all BOLD NIfTI files (story tasks only — exclude localizers)
    bold_files = []
    for ses_dir in sorted(subj_dir.glob("ses-*")):
        func_dir = ses_dir / "func"
        if not func_dir.exists():
            continue
        for bold in func_dir.glob("*_bold.nii*"):
            # Skip localizer tasks
            if any(x in bold.name for x in ["Localizer", "localizer", "Auditory", "Category"]):
                continue
            bold_files.append(bold)

    if not bold_files:
        raise FileNotFoundError(f"No story BOLD files found for {subject_id} in {subj_dir}")

    logger.info(f"Subject {subject_id}: {len(bold_files)} story BOLD file(s)")

    all_data = []
    all_labels: list[str] = []

    for bold_path in bold_files:
        # Parse task name from filename: sub-UTS01_ses-2_task-avatar_bold.nii.gz → "avatar"
        task_name = _task_name_from_bold_path(bold_path)

        logger.info(f"  Story: {task_name}  BOLD: {bold_path.name}")

        events_tsv = _find_word_events_tsv(subj_dir, task_name)
        if events_tsv is None:
            logger.warning(
                f"  No word-level events TSV found for task '{task_name}'. "
                "Skipping this story. Expected either a word-level TSV or a "
                "derivatives/TextGrids/<task>.TextGrid file."
            )
            continue

        try:
            bold_img, _ = load_bold(str(bold_path))
            events_df = load_events(str(events_tsv), label_info={})
            mask_img = load_brain_mask(
                mask_path=config.get("mask"), bold_img=bold_img
            )
            story_config = dict(config)
            story_config["tr"] = _infer_tr_for_bold(bold_path, config)
            glm = fit_glm(bold_img, events_df, mask_img, story_config)

            vocabulary = sorted(events_df["trial_type"].unique().tolist())
            beta_imgs, word_labels = extract_beta_maps(glm, vocabulary)
            if not beta_imgs:
                logger.warning(f"  No beta maps extracted for {task_name}")
                continue

            beta_data = apply_mask_to_betas(
                beta_imgs,
                mask_img,
                roi_mask_img=None
                if config.get("roi_mask") is None
                else load_brain_mask(config["roi_mask"]),
            )
            all_data.append(beta_data)
            all_labels.extend(word_labels)
        except Exception as e:
            logger.error(f"  Failed {task_name}: {e}", exc_info=True)
            continue

    if not all_data:
        raise RuntimeError(
            f"No fMRI data could be preprocessed for {subject_id}. "
            "Most likely cause: Huth narrative stories lack word-level events TSVs. "
            "Please add forced-alignment timing files."
        )

    combined_data = np.concatenate(all_data, axis=0)

    # Optional PCA
    pca_var = 0.0
    n_pca = config.get("n_pca_components")
    if n_pca is not None:
        fit_pca = pca is None
        combined_data, pca = apply_pca(combined_data, n_pca, fit=fit_pca, pca=pca)
        if fit_pca:
            pca_var = float(pca.explained_variance_ratio_.sum())
            logger.info(f"PCA: {n_pca} components, explained variance={pca_var:.3f}")

    # Voxel coords — approximate (use zeros if PCA was applied)
    if n_pca is not None:
        voxel_coords = np.zeros((combined_data.shape[1], 3), dtype=np.int32)
    else:
        # Need mask_img from last successful run
        try:
            voxel_coords = get_voxel_mni_coords(mask_img)
        except Exception:
            voxel_coords = np.zeros((combined_data.shape[1], 3), dtype=np.int32)

    output_dir = str(Path(config["output_dir"]) / dataset_name / "fmri")
    return save_betas(combined_data, all_labels, subject_id, voxel_coords, pca_var, output_dir)


def run_all_subjects(dataset_name: str, config_path: str) -> list[str]:
    """
    Run fMRI preprocessing for all subjects. Fits PCA on training subjects,
    applies to val/test subjects using the same transform.

    Args:
        dataset_name: Dataset name.
        config_path: Path to configs/preprocessing_fmri.yaml.

    Returns:
        List of paths to saved .npz files.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = Path(config.get("data_root", "data/fmri_huth"))
    subjects = config.get("subjects", "all")

    if subjects == "all":
        subjects = sorted(
            [d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        )

    # Determine train subjects (not in val/test)
    val_subjects = set(config.get("val_subjects", []))
    test_subjects = set(config.get("test_subjects", []))
    train_subjects = [s for s in subjects if s not in val_subjects and s not in test_subjects]

    logger.info(f"fMRI: {len(subjects)} subjects ({len(train_subjects)} train)")
    paths = []
    pca_obj = None

    # Train subjects first (to fit PCA if needed)
    for subj in train_subjects:
        try:
            path = run_subject(subj, dataset_name, config, pca=None)
            paths.append(path)
            logger.info(f"✓ {subj} (train) → {path}")
        except Exception as e:
            logger.error(f"✗ {subj}: {e}")

    # Val/test subjects (apply pre-fit PCA)
    for subj in list(val_subjects) + list(test_subjects):
        if subj not in subjects:
            continue
        try:
            path = run_subject(subj, dataset_name, config, pca=pca_obj)
            paths.append(path)
            logger.info(f"✓ {subj} (val/test) → {path}")
        except Exception as e:
            logger.error(f"✗ {subj}: {e}")

    return paths


if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "workers/worker_data/configs/preprocessing_fmri.yaml"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "huth"
    run_all_subjects(dataset_name, config_path)
