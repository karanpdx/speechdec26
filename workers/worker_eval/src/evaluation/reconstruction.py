"""
Stage 2 evaluation: neural signal reconstruction quality metrics.

All functions operate on numpy arrays — no PyTorch or model dependencies.

Functions:
    compute_n400_correlation        — ERP component recovery (EEG/MEG)
    compute_fmri_spatial_correlation — voxel-level correlation
    compute_neurosynth_correlation   — language map comparison
    compute_round_trip_similarity    — embedding consistency check
    plot_erp_comparison              — visualization
    plot_fmri_glass_brain            — visualization
    generate_stage2_report           — write evaluation/stage2_report.md
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_erp(
    epochs: np.ndarray,
    sfreq: float,
    tmin: float = -0.1,
) -> np.ndarray:
    """
    Compute grand average ERP by averaging across epochs.

    Args:
        epochs: float32 (n_epochs, n_channels, n_timepoints)
        sfreq:  sampling frequency in Hz
        tmin:   epoch start time relative to word onset (s) — for time axis

    Returns:
        float32 (n_channels, n_timepoints) — grand average ERP
    """
    raise NotImplementedError


def compute_n400_amplitude(
    erp: np.ndarray,
    sfreq: float,
    tmin: float = -0.1,
    n400_window: tuple = (0.3, 0.5),
    baseline_window: tuple = (-0.1, 0.0),
) -> float:
    """
    Compute N400 amplitude at centro-parietal channels.

    N400 = mean amplitude in n400_window minus mean amplitude in baseline_window.
    Computed at centro-parietal channels (Pz, CPz, or channels at indices
    matching those names in ch_names).

    Args:
        erp:             float32 (n_channels, n_timepoints) grand average ERP
        sfreq:           sampling frequency in Hz
        tmin:            epoch start time in seconds
        n400_window:     (start, end) in seconds for N400 measurement
        baseline_window: (start, end) in seconds for baseline

    Returns:
        float — N400 amplitude (negative values indicate N400 effect)
    """
    raise NotImplementedError


def compute_n400_correlation(
    pred_epochs: np.ndarray,
    real_epochs: np.ndarray,
    word_labels: list,
    ch_names: list,
    sfreq: float,
    tmin: float = -0.1,
) -> dict:
    """
    Compute Pearson correlation between real and predicted N400 amplitudes across words.

    For each unique word:
        1. Average epochs for that word (real and predicted separately)
        2. Compute N400 amplitude for each

    Then correlate the per-word N400 amplitudes.

    Args:
        pred_epochs:  float32 (n_samples, n_channels, n_timepoints)
        real_epochs:  float32 (n_samples, n_channels, n_timepoints)
        word_labels:  list[str] length n_samples
        ch_names:     list[str] length n_channels
        sfreq:        sampling frequency
        tmin:         epoch start time in seconds

    Returns:
        Dict with keys:
            'pearson_r':      float — correlation coefficient
            'p_value':        float — two-tailed p-value
            'n_words':        int — number of unique words
            'pred_n400':      np.ndarray (n_words,) — predicted N400 per word
            'real_n400':      np.ndarray (n_words,) — real N400 per word
            'word_labels':    list[str]
    """
    raise NotImplementedError


def plot_erp_comparison(
    pred_erp: np.ndarray,
    real_erp: np.ndarray,
    ch_names: list,
    sfreq: float,
    tmin: float = -0.1,
    output_path: str = "evaluation/figures/erp_comparison.png",
) -> str:
    """
    Plot real vs. predicted ERP waveforms at centro-parietal channels.

    Args:
        pred_erp:    float32 (n_channels, n_timepoints)
        real_erp:    float32 (n_channels, n_timepoints)
        ch_names:    list[str]
        sfreq:       sampling frequency in Hz
        tmin:        epoch start time in seconds
        output_path: where to save the figure

    Returns:
        Path to saved figure.
    """
    raise NotImplementedError


def compute_fmri_spatial_correlation(
    pred_betas: np.ndarray,
    real_betas: np.ndarray,
) -> dict:
    """
    Compute Pearson correlation between predicted and real beta maps across voxels,
    per word. Reports mean ± std across words.

    Args:
        pred_betas: float32 (n_words, n_voxels) — predicted beta maps
        real_betas: float32 (n_words, n_voxels) — real beta maps

    Returns:
        Dict with keys:
            'per_word_r':  np.ndarray (n_words,) — per-word correlation
            'mean_r':      float
            'std_r':       float
    """
    raise NotImplementedError


def compute_neurosynth_correlation(
    pred_mean_betas: np.ndarray,
    voxel_coords: np.ndarray,
) -> dict:
    """
    Correlate mean predicted beta map with Neurosynth language localizer.

    Fetches the Neurosynth language map using nilearn and correlates
    with the predicted mean activation.

    Args:
        pred_mean_betas: float32 (n_voxels,) — mean predicted activation across words
        voxel_coords:    int32 (n_voxels, 3) — MNI coordinates

    Returns:
        Dict with keys:
            'pearson_r':  float — correlation with language localizer
            'p_value':    float
    """
    raise NotImplementedError


def plot_fmri_glass_brain(
    mean_betas: np.ndarray,
    voxel_coords: np.ndarray,
    output_path: str = "evaluation/figures/fmri_glass_brain.png",
) -> str:
    """
    Produce a glass brain visualization of mean predicted activation.

    Uses nilearn.plotting.plot_glass_brain.

    Args:
        mean_betas:   float32 (n_voxels,) — mean activation to visualize
        voxel_coords: int32 (n_voxels, 3) — MNI coordinates
        output_path:  where to save figure

    Returns:
        Path to saved figure.
    """
    raise NotImplementedError


def compute_round_trip_similarity(
    original_embeddings: np.ndarray,
    round_trip_embeddings: np.ndarray,
) -> dict:
    """
    Compute cosine similarity between original and round-trip embeddings.

    Round-trip: BERT → SharedEmbeddingProjector → decoder → encoder → shared space.
    A mean similarity > 0.5 indicates basic pipeline consistency.
    A mean similarity > 0.7 indicates strong internal consistency.

    Args:
        original_embeddings:   float32 (n_words, embed_dim) — original shared embeddings
        round_trip_embeddings: float32 (n_words, embed_dim) — after encoder/decoder cycle

    Returns:
        Dict with keys:
            'per_word_similarity': np.ndarray (n_words,)
            'mean':                float
            'std':                 float
            'fraction_above_0.5':  float
            'fraction_above_0.7':  float
    """
    raise NotImplementedError


def generate_stage2_report(
    n400_metrics: dict,
    fmri_spatial_metrics: dict,
    neurosynth_metrics: dict,
    round_trip_metrics: dict,
    good_examples: list,
    failure_examples: list,
    figure_paths: dict,
    output_path: str = "evaluation/stage2_report.md",
) -> str:
    """
    Write the Stage 2 evaluation report to markdown.

    Sections:
        1. N400 correlation: Pearson r, p-value
        2. fMRI spatial correlation: mean ± std across words
        3. Neurosynth language map correlation
        4. Round-trip cosine similarity: mean ± std, fraction > 0.5 and > 0.7
        5. 5 good reconstruction examples
        6. 5 failure examples

    Args:
        n400_metrics:       Output of compute_n400_correlation()
        fmri_spatial_metrics: Output of compute_fmri_spatial_correlation()
        neurosynth_metrics: Output of compute_neurosynth_correlation()
        round_trip_metrics: Output of compute_round_trip_similarity()
        good_examples:      list of dicts {'word', 'real_n400', 'pred_n400'}
        failure_examples:   same format
        figure_paths:       dict {'erp': str, 'glass_brain': str}
        output_path:        where to write the report

    Returns:
        Path to written report.
    """
    raise NotImplementedError
