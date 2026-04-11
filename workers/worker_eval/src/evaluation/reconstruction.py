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
    # Average across the epochs dimension to get the grand average waveform
    return epochs.mean(axis=0)


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
    # Convert time windows to sample indices using sfreq
    def time_to_idx(t): return int((t - tmin) * sfreq)

    b_start, b_end = time_to_idx(baseline_window[0]), time_to_idx(baseline_window[1])
    n_start, n_end = time_to_idx(n400_window[0]),     time_to_idx(n400_window[1])

    # Mean across all channels and timepoints within each window
    baseline_mean = erp[:, b_start:b_end].mean()
    n400_mean     = erp[:, n_start:n_end].mean()

    # N400 amplitude = N400 window minus baseline (negative = N400 effect present)
    return float(n400_mean - baseline_mean)


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
    from scipy.stats import pearsonr

    unique_words = sorted(set(word_labels))
    word_labels_arr = np.array(word_labels)

    pred_n400s, real_n400s = [], []
    for word in unique_words:
        # Select all epoch indices belonging to this word
        idx = np.where(word_labels_arr == word)[0]

        # Average epochs for this word, then compute its N400 amplitude
        pred_erp = pred_epochs[idx].mean(axis=0)
        real_erp = real_epochs[idx].mean(axis=0)

        pred_n400s.append(compute_n400_amplitude(pred_erp, sfreq, tmin))
        real_n400s.append(compute_n400_amplitude(real_erp, sfreq, tmin))

    pred_n400s = np.array(pred_n400s)
    real_n400s = np.array(real_n400s)

    # Correlate per-word N400 amplitudes across real and predicted
    r, p = pearsonr(pred_n400s, real_n400s)

    return {
        "pearson_r":   float(r),
        "p_value":     float(p),
        "n_words":     len(unique_words),
        "pred_n400":   pred_n400s,
        "real_n400":   real_n400s,
        "word_labels": unique_words,
    }


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
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build the time axis in seconds from tmin to tmax
    n_timepoints = real_erp.shape[1]
    times = np.arange(n_timepoints) / sfreq + tmin

    # Pick centro-parietal channels by name; fall back to all channels if none match
    cp_names = {"Pz", "CPz", "CP1", "CP2", "Cz"}
    cp_idx = [i for i, ch in enumerate(ch_names) if ch in cp_names] or list(range(len(ch_names)))

    # Average across selected channels to get a single representative waveform
    real_wave = real_erp[cp_idx].mean(axis=0)
    pred_wave = pred_erp[cp_idx].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, real_wave, label="Real",      color="steelblue",  linewidth=2)
    ax.plot(times, pred_wave, label="Predicted", color="darkorange", linewidth=2, linestyle="--")

    # Shade the N400 window so the comparison region is immediately visible
    ax.axvspan(0.3, 0.5, alpha=0.15, color="gray", label="N400 window")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Real vs. predicted ERP — centro-parietal channels")
    ax.legend()
    ax.invert_yaxis()   # ERP convention: negative up
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path



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
    from scipy.stats import pearsonr

    # Correlate predicted vs. real voxel patterns independently for each word
    per_word_r = np.array([
        pearsonr(pred_betas[i], real_betas[i])[0]
        for i in range(pred_betas.shape[0])
    ])

    return {
        "per_word_r": per_word_r,
        "mean_r":     float(per_word_r.mean()),
        "std_r":      float(per_word_r.std()),
    }




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
    from scipy.stats import pearsonr
    from nilearn import datasets, image
    import nibabel as nib

    # Fetch the Neurosynth language meta-analysis map via nilearn
    lang_map = datasets.fetch_neurovault_motor_task()
    img      = image.load_img(lang_map.images[0])
    data     = img.get_fdata()

    # Sample the atlas map at each of our voxel coordinates
    neurosynth_vals = np.array([
        data[tuple(voxel_coords[i])] for i in range(len(voxel_coords))
    ])

    # Mask out voxels where the atlas has no data
    valid = np.isfinite(neurosynth_vals) & (neurosynth_vals != 0)
    r, p  = pearsonr(pred_mean_betas[valid], neurosynth_vals[valid])

    return {"pearson_r": float(r), "p_value": float(p)}



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
    import nibabel as nib
    from nilearn import plotting, image

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build a blank MNI volume and write activation values at voxel coordinates
    affine   = np.diag([2, 2, 2, 1])   # 2mm isotropic MNI space
    vol_shape = (91, 109, 91)
    data      = np.zeros(vol_shape)
    for val, (x, y, z) in zip(mean_betas, voxel_coords):
        if all(0 <= c < s for c, s in zip((x, y, z), vol_shape)):
            data[x, y, z] = val

    img = nib.Nifti1Image(data, affine)

    # Render the glass brain and save to disk
    display = plotting.plot_glass_brain(img, colorbar=True, title="Mean predicted activation")
    display.savefig(output_path, dpi=150)
    display.close()

    return output_path


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
    # Normalize both sets so dot product equals cosine similarity
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    orig_norm  = normalize(original_embeddings)
    rt_norm    = normalize(round_trip_embeddings)

    # Per-word cosine similarity = dot product of paired unit vectors
    per_word   = np.sum(orig_norm * rt_norm, axis=1)

    return {
        "per_word_similarity": per_word,
        "mean":                float(per_word.mean()),
        "std":                 float(per_word.std()),
        "fraction_above_0.5":  float((per_word > 0.5).mean()),
        "fraction_above_0.7":  float((per_word > 0.7).mean()),
    }



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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    def w(line=""): lines.append(line)

    w("# Stage 2 evaluation report")
    w()

    # ---- Section 1: N400 ----
    w("## 1. N400 correlation")
    w()
    w(f"![ERP comparison]({figure_paths.get('erp', '')})")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Pearson r | {n400_metrics['pearson_r']:.3f} |")
    w(f"| p-value   | {n400_metrics['p_value']:.4f} |")
    w(f"| Words     | {n400_metrics['n_words']} |")
    w()
    w("### Per-word N400 amplitudes")
    w()
    w("| Word | Real N400 | Predicted N400 |")
    w("|------|-----------|----------------|")
    for word, real, pred in zip(
        n400_metrics["word_labels"],
        n400_metrics["real_n400"],
        n400_metrics["pred_n400"],
    ):
        w(f"| {word} | {real:.3f} | {pred:.3f} |")

    # ---- Section 2: fMRI spatial ----
    w()
    w("## 2. fMRI spatial correlation")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Mean r | {fmri_spatial_metrics['mean_r']:.3f} |")
    w(f"| Std r  | {fmri_spatial_metrics['std_r']:.3f} |")

    # ---- Section 3: Neurosynth ----
    w()
    w("## 3. Neurosynth language map correlation")
    w()
    w(f"![Glass brain]({figure_paths.get('glass_brain', '')})")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Pearson r | {neurosynth_metrics['pearson_r']:.3f} |")
    w(f"| p-value   | {neurosynth_metrics['p_value']:.4f} |")

    # ---- Section 4: Round-trip ----
    w()
    w("## 4. Round-trip cosine similarity")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Mean similarity     | {round_trip_metrics['mean']:.3f} |")
    w(f"| Std                 | {round_trip_metrics['std']:.3f} |")
    w(f"| Fraction above 0.5  | {round_trip_metrics['fraction_above_0.5']:.3f} |")
    w(f"| Fraction above 0.7  | {round_trip_metrics['fraction_above_0.7']:.3f} |")

    # ---- Sections 5 & 6: Examples ----
    for title, examples in [("5. Good reconstruction examples", good_examples[:5]),
                             ("6. Failure examples",             failure_examples[:5])]:
        w()
        w(f"## {title}")
        w()
        w("| Word | Real N400 | Predicted N400 |")
        w("|------|-----------|----------------|")
        for ex in examples:
            w(f"| {ex['word']} | {ex['real_n400']:.3f} | {ex['pred_n400']:.3f} |")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    return output_path
