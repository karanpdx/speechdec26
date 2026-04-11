"""
Stage 1 evaluation: word retrieval metrics.

All functions operate on numpy arrays — no PyTorch or model dependencies.

Functions:
    compute_retrieval_metrics           — top-k accuracy and MRR
    compute_cross_subject_generalization — per-subject retrieval metrics
    compute_cross_modal_alignment        — EEG/MEG embedding alignment gap
    compute_abstention_curve             — coverage/accuracy tradeoff
    generate_stage1_report               — write evaluation/stage1_report.md
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(
    neural_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list,
    vocab: list,
    k_values: list = (1, 5, 10),
) -> dict:
    """
    Compute cosine retrieval accuracy (top-k) and mean reciprocal rank.

    For each sample: compute cosine similarity against all vocab embeddings,
    rank descending, check if ground truth is in top-k.
    MRR = mean of 1/rank across all samples.

    Args:
        neural_embeddings: float32 (n_samples, embed_dim)
        text_embeddings:   float32 (vocab_size, embed_dim)
        labels:            list[str] length n_samples — ground truth words
        vocab:             list[str] length vocab_size
        k_values:          list of k values to compute accuracy for

    Returns:
        Dict with keys 'top{k}' for each k in k_values, plus 'mrr'.
        Example: {'top1': 0.25, 'top5': 0.60, 'top10': 0.75, 'mrr': 0.38}
    """
    raise NotImplementedError


def compute_cross_subject_generalization(
    neural_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list,
    vocab: list,
    subject_ids: list,
    k_values: list = (1, 5, 10),
) -> dict:
    """
    Compute per-subject retrieval metrics and mean across subjects.

    IMPORTANT: Reports mean across subjects (not across pooled samples).
    Pooling inflates performance for subjects with more samples.

    Args:
        neural_embeddings: float32 (n_samples, embed_dim)
        text_embeddings:   float32 (vocab_size, embed_dim)
        labels:            list[str] length n_samples
        vocab:             list[str] length vocab_size
        subject_ids:       list[str] length n_samples

    Returns:
        Dict with keys:
            'per_subject': dict mapping subject_id → metrics dict
            'mean': metrics dict averaged across subjects
            'std': metrics dict std across subjects
    """
    raise NotImplementedError


def compute_cross_modal_alignment(
    eeg_embeddings: np.ndarray,
    meg_embeddings: np.ndarray,
    eeg_labels: list,
    meg_labels: list,
) -> dict:
    """
    Measure how well the shared space aligns EEG and MEG embeddings.

    For matched word pairs (same word label in both modalities):
    - Compute cosine similarity between EEG and MEG embeddings
    Compare to:
    - Cosine similarity between randomly paired embeddings (negative control)

    A positive gap (matched >> random) confirms genuine cross-modal alignment.

    Args:
        eeg_embeddings: float32 (n_eeg, embed_dim)
        meg_embeddings: float32 (n_meg, embed_dim)
        eeg_labels:     list[str] length n_eeg
        meg_labels:     list[str] length n_meg

    Returns:
        Dict with keys:
            'matched_similarity': float — mean cosine sim of matched pairs
            'random_similarity':  float — mean cosine sim of random pairs
            'alignment_gap':      float — matched - random (must be > 0)
            'n_matched_pairs':    int   — number of matched word pairs found
    """
    raise NotImplementedError


def compute_abstention_curve(
    neural_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list,
    vocab: list,
    confidence_thresholds: np.ndarray = None,
) -> dict:
    """
    Compute coverage/accuracy tradeoff under abstention.

    For each threshold t:
    - Abstain when (top1_score - top2_score) < t
    - Compute accuracy on non-abstained samples
    - Compute coverage = fraction of samples not abstained

    Args:
        neural_embeddings: float32 (n_samples, embed_dim)
        text_embeddings:   float32 (vocab_size, embed_dim)
        labels:            list[str] length n_samples
        vocab:             list[str] length vocab_size
        confidence_thresholds: 1D array of thresholds to sweep.
                               Defaults to np.linspace(0, 1, 50).

    Returns:
        Dict with keys:
            'thresholds':  1D array
            'coverage':    1D array — fraction not abstained at each threshold
            'accuracy':    1D array — accuracy on non-abstained at each threshold
            'best_threshold_80pct': float — threshold achieving 80% accuracy
                                    with maximum coverage (or None if 80% unreachable)
    """
    raise NotImplementedError


def generate_stage1_report(
    test_metrics: dict,
    cross_subject_metrics: dict,
    alignment_metrics: dict,
    abstention_metrics: dict,
    failure_cases: list,
    output_path: str = "evaluation/stage1_report.md",
) -> str:
    """
    Write the Stage 1 evaluation report to markdown.

    Sections:
        1. Per-modality top-1, top-5, top-10, MRR
        2. Cross-subject generalization (per-subject table, mean ± std)
        3. Cross-modal alignment (matched vs. random similarity)
        4. Abstention curve (coverage/accuracy table)
        5. 10 representative failure cases

    Args:
        test_metrics:          Output of compute_retrieval_metrics() — dict per modality
        cross_subject_metrics: Output of compute_cross_subject_generalization()
        alignment_metrics:     Output of compute_cross_modal_alignment()
        abstention_metrics:    Output of compute_abstention_curve()
        failure_cases:         List of dicts: {'true_label', 'top3_predictions', 'scores'}
        output_path:           Where to write the report.

    Returns:
        Path to written report.
    """
    raise NotImplementedError
