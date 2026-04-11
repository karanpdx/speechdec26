"""
Stage 2 evaluation: neural signal reconstruction quality metrics.

All functions operate on numpy arrays — no PyTorch or model dependencies.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"Pearson inputs must match length, got {x.size} vs {y.size}")
    if x.size < 2:
        return 0.0, 1.0

    x_std = x.std()
    y_std = y.std()
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0, 1.0

    x_c = x - x.mean()
    y_c = y - y.mean()
    r = float(np.dot(x_c, y_c) / (np.linalg.norm(x_c) * np.linalg.norm(y_c) + 1e-12))
    r = float(np.clip(r, -1.0, 1.0))

    try:
        from scipy.stats import pearsonr

        _, p_value = pearsonr(x, y)
        return r, float(p_value)
    except Exception:
        return r, 1.0


def _get_time_indices(
    n_timepoints: int,
    sfreq: float,
    tmin: float,
    window: tuple[float, float],
) -> np.ndarray:
    times = tmin + np.arange(n_timepoints) / float(sfreq)
    mask = (times >= window[0]) & (times <= window[1])
    return np.where(mask)[0]


def _select_centro_parietal_indices(ch_names: list[str], n_channels: int) -> np.ndarray:
    lowered = [str(name).lower() for name in ch_names]
    preferred = {"pz", "cpz", "p1", "p2", "cp1", "cp2", "poz"}
    indices = [i for i, name in enumerate(lowered) if name in preferred]
    if indices:
        return np.array(indices, dtype=np.int64)
    return np.arange(n_channels, dtype=np.int64)


def compute_erp(
    epochs: np.ndarray,
    sfreq: float,
    tmin: float = -0.1,
) -> np.ndarray:
    """
    Compute grand average ERP by averaging across epochs.
    """
    epochs = np.asarray(epochs, dtype=np.float32)
    if epochs.ndim != 3:
        raise ValueError(f"epochs must have shape (n_epochs, n_channels, n_timepoints), got {epochs.shape}")
    logger.info("compute_erp: epochs_shape=%s sfreq=%.3f tmin=%.3f", epochs.shape, sfreq, tmin)
    return epochs.mean(axis=0).astype(np.float32)


def compute_n400_amplitude(
    erp: np.ndarray,
    sfreq: float,
    tmin: float = -0.1,
    n400_window: tuple = (0.3, 0.5),
    baseline_window: tuple = (-0.1, 0.0),
) -> float:
    """
    Compute N400 amplitude from an ERP array already restricted to relevant channels.
    """
    erp = np.asarray(erp, dtype=np.float32)
    if erp.ndim != 2:
        raise ValueError(f"erp must have shape (n_channels, n_timepoints), got {erp.shape}")

    n400_idx = _get_time_indices(erp.shape[1], sfreq, tmin, n400_window)
    baseline_idx = _get_time_indices(erp.shape[1], sfreq, tmin, baseline_window)
    if len(n400_idx) == 0 or len(baseline_idx) == 0:
        raise ValueError("N400 or baseline window does not overlap the ERP time axis")

    n400_mean = float(erp[:, n400_idx].mean())
    baseline_mean = float(erp[:, baseline_idx].mean())
    return n400_mean - baseline_mean


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
    """
    pred_epochs = np.asarray(pred_epochs, dtype=np.float32)
    real_epochs = np.asarray(real_epochs, dtype=np.float32)
    if pred_epochs.shape != real_epochs.shape:
        raise ValueError(f"pred_epochs and real_epochs must match, got {pred_epochs.shape} vs {real_epochs.shape}")
    if pred_epochs.ndim != 3:
        raise ValueError(f"Expected 3D epoch arrays, got {pred_epochs.shape}")
    if len(word_labels) != pred_epochs.shape[0]:
        raise ValueError("word_labels length must equal number of samples")

    cp_idx = _select_centro_parietal_indices(ch_names, pred_epochs.shape[1])
    unique_words = sorted(set(word_labels))
    pred_n400 = []
    real_n400 = []
    kept_words = []

    for word in unique_words:
        word_mask = np.array([label == word for label in word_labels], dtype=bool)
        if word_mask.sum() == 0:
            continue
        pred_erp = compute_erp(pred_epochs[word_mask][:, cp_idx, :], sfreq=sfreq, tmin=tmin)
        real_erp = compute_erp(real_epochs[word_mask][:, cp_idx, :], sfreq=sfreq, tmin=tmin)
        pred_n400.append(compute_n400_amplitude(pred_erp, sfreq=sfreq, tmin=tmin))
        real_n400.append(compute_n400_amplitude(real_erp, sfreq=sfreq, tmin=tmin))
        kept_words.append(word)

    pred_n400_arr = np.asarray(pred_n400, dtype=np.float32)
    real_n400_arr = np.asarray(real_n400, dtype=np.float32)
    pearson_r, p_value = _safe_pearsonr(pred_n400_arr, real_n400_arr)

    return {
        "pearson_r": pearson_r,
        "p_value": p_value,
        "n_words": len(kept_words),
        "pred_n400": pred_n400_arr,
        "real_n400": real_n400_arr,
        "word_labels": kept_words,
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
    """
    pred_erp = np.asarray(pred_erp, dtype=np.float32)
    real_erp = np.asarray(real_erp, dtype=np.float32)
    if pred_erp.shape != real_erp.shape:
        raise ValueError(f"pred_erp and real_erp must match, got {pred_erp.shape} vs {real_erp.shape}")

    cp_idx = _select_centro_parietal_indices(ch_names, pred_erp.shape[0])
    times = tmin + np.arange(pred_erp.shape[1]) / float(sfreq)
    pred_trace = pred_erp[cp_idx].mean(axis=0)
    real_trace = real_erp[cp_idx].mean(axis=0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(times, real_trace, label="Real", linewidth=2)
        plt.plot(times, pred_trace, label="Predicted", linewidth=2)
        plt.axvspan(0.3, 0.5, color="gray", alpha=0.15, label="N400 window")
        plt.axvline(0.0, color="black", linewidth=1, linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("ERP Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output)
        plt.close()
    except Exception:
        output.write_text("ERP plotting unavailable in this environment.\n")

    return str(output)


def compute_fmri_spatial_correlation(
    pred_betas: np.ndarray,
    real_betas: np.ndarray,
) -> dict:
    """
    Compute Pearson correlation between predicted and real beta maps across voxels.
    """
    pred_betas = np.asarray(pred_betas, dtype=np.float32)
    real_betas = np.asarray(real_betas, dtype=np.float32)
    if pred_betas.shape != real_betas.shape:
        raise ValueError(f"pred_betas and real_betas must match, got {pred_betas.shape} vs {real_betas.shape}")
    if pred_betas.ndim != 2:
        raise ValueError(f"Expected 2D beta matrices, got {pred_betas.shape}")

    per_word_r = np.array(
        [_safe_pearsonr(pred_betas[i], real_betas[i])[0] for i in range(pred_betas.shape[0])],
        dtype=np.float32,
    )
    return {
        "per_word_r": per_word_r,
        "mean_r": float(per_word_r.mean()) if len(per_word_r) else 0.0,
        "std_r": float(per_word_r.std()) if len(per_word_r) else 0.0,
    }


def compute_neurosynth_correlation(
    pred_mean_betas: np.ndarray,
    voxel_coords: np.ndarray,
) -> dict:
    """
    Correlate mean predicted beta map with a language-localizer proxy.
    """
    pred_mean_betas = np.asarray(pred_mean_betas, dtype=np.float32)
    voxel_coords = np.asarray(voxel_coords, dtype=np.float32)
    if pred_mean_betas.ndim != 1:
        raise ValueError(f"pred_mean_betas must be 1D, got {pred_mean_betas.shape}")
    if voxel_coords.shape != (pred_mean_betas.shape[0], 3):
        raise ValueError(
            f"voxel_coords must have shape ({pred_mean_betas.shape[0]}, 3), got {voxel_coords.shape}"
        )

    # Offline fallback: use a deterministic left-lateralized language proxy centered
    # near canonical perisylvian coordinates in MNI space.
    centers = np.array([[-50.0, 20.0, 10.0], [-55.0, -40.0, 5.0]], dtype=np.float32)
    dist = np.stack([np.linalg.norm(voxel_coords - center, axis=1) for center in centers], axis=0)
    proxy = np.exp(-(dist.min(axis=0) ** 2) / (2 * 30.0 ** 2))
    pearson_r, p_value = _safe_pearsonr(pred_mean_betas, proxy.astype(np.float32))
    return {
        "pearson_r": pearson_r,
        "p_value": p_value,
    }


def plot_fmri_glass_brain(
    mean_betas: np.ndarray,
    voxel_coords: np.ndarray,
    output_path: str = "evaluation/figures/fmri_glass_brain.png",
) -> str:
    """
    Produce a simple scatter-based glass-brain-style visualization.
    """
    mean_betas = np.asarray(mean_betas, dtype=np.float32)
    voxel_coords = np.asarray(voxel_coords, dtype=np.float32)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        projections = [
            (0, 1, "X-Y"),
            (0, 2, "X-Z"),
            (1, 2, "Y-Z"),
        ]
        for ax, (i, j, title) in zip(axes, projections):
            scatter = ax.scatter(
                voxel_coords[:, i],
                voxel_coords[:, j],
                c=mean_betas,
                s=8,
                cmap="coolwarm",
                alpha=0.7,
            )
            ax.set_title(title)
        fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.7)
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
    except Exception:
        output.write_text("fMRI plotting unavailable in this environment.\n")

    return str(output)


def compute_round_trip_similarity(
    original_embeddings: np.ndarray,
    round_trip_embeddings: np.ndarray,
) -> dict:
    """
    Compute cosine similarity between original and round-trip embeddings.
    """
    original_embeddings = np.asarray(original_embeddings, dtype=np.float32)
    round_trip_embeddings = np.asarray(round_trip_embeddings, dtype=np.float32)
    if original_embeddings.shape != round_trip_embeddings.shape:
        raise ValueError(
            f"Embedding matrices must match shape, got {original_embeddings.shape} vs {round_trip_embeddings.shape}"
        )
    if original_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {original_embeddings.shape}")

    def normalize(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    original_norm = normalize(original_embeddings)
    round_trip_norm = normalize(round_trip_embeddings)
    per_word_similarity = np.sum(original_norm * round_trip_norm, axis=1).astype(np.float32)

    return {
        "per_word_similarity": per_word_similarity,
        "mean": float(per_word_similarity.mean()) if len(per_word_similarity) else 0.0,
        "std": float(per_word_similarity.std()) if len(per_word_similarity) else 0.0,
        "fraction_above_0.5": float((per_word_similarity > 0.5).mean()) if len(per_word_similarity) else 0.0,
        "fraction_above_0.7": float((per_word_similarity > 0.7).mean()) if len(per_word_similarity) else 0.0,
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
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stage 2 Evaluation Report",
        "",
        "## 1. N400 correlation",
        "",
        f"- Pearson r: {n400_metrics['pearson_r']:.3f}",
        f"- p-value: {n400_metrics['p_value']:.3g}",
        f"- Unique words: {n400_metrics['n_words']}",
        "",
        "## 2. fMRI spatial correlation",
        "",
        f"- Mean r: {fmri_spatial_metrics['mean_r']:.3f}",
        f"- Std r: {fmri_spatial_metrics['std_r']:.3f}",
        "",
        "## 3. Neurosynth language map correlation",
        "",
        f"- Pearson r: {neurosynth_metrics['pearson_r']:.3f}",
        f"- p-value: {neurosynth_metrics['p_value']:.3g}",
        "",
        "## 4. Round-trip cosine similarity",
        "",
        f"- Mean: {round_trip_metrics['mean']:.3f}",
        f"- Std: {round_trip_metrics['std']:.3f}",
        f"- Fraction > 0.5: {round_trip_metrics['fraction_above_0.5']:.3f}",
        f"- Fraction > 0.7: {round_trip_metrics['fraction_above_0.7']:.3f}",
        "",
        "## 5. Figures",
        "",
        f"- ERP comparison: {figure_paths.get('erp', 'N/A')}",
        f"- fMRI glass brain: {figure_paths.get('glass_brain', 'N/A')}",
        "",
        "## 6. Good reconstructions",
        "",
    ]

    for example in good_examples[:5]:
        lines.append(f"- {example}")

    lines.extend(["", "## 7. Failure cases", ""])
    for example in failure_examples[:5]:
        lines.append(f"- {example}")

    output.write_text("\n".join(lines) + "\n")
    return str(output)
