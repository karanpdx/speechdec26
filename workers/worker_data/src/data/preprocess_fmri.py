"""
fMRI preprocessing pipeline.

Transforms raw BOLD NIfTI timeseries into per-word beta maps via a GLM
with HRF convolution.

Output schema (saved as .npz):
    data:                   float32  (n_words, n_voxels)
    labels:                 list[str]  length n_words
    subject_id:             str
    voxel_coords:           int32    (n_voxels, 3)  MNI coordinates
    pca_explained_variance: float    (0.0 if PCA not applied)
"""

import logging

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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def extract_beta_maps(glm, vocabulary: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Extract a beta map (effect size contrast) per unique word.

    Args:
        glm: Fitted nilearn FirstLevelModel.
        vocabulary: List of unique word strings to extract betas for.

    Returns:
        Tuple of (beta_maps nibabel image list, word_labels list[str]).
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def run_subject(subject_id: str, dataset_name: str, config: dict, pca=None) -> str:
    """
    Run full fMRI preprocessing pipeline for one subject.

    Args:
        subject_id: Subject identifier.
        dataset_name: Dataset name.
        config: Parsed preprocessing_fmri.yaml config.
        pca: Pre-fit PCA object for val/test subjects, or None for training subjects.

    Returns:
        Path to saved .npz file.
    """
    raise NotImplementedError


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
    raise NotImplementedError
