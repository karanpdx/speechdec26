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
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

def _validate_retrieval_inputs(neural_embeddings, text_embeddings, labels, vocab):
    if neural_embeddings.ndim != 2:
        raise ValueError("neural_embeddings must be 2D: (n_samples, embed_dim)")
    if text_embeddings.ndim != 2:
        raise ValueError("text_embeddings must be 2D: (vocab_size, embed_dim)")
    if neural_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("Embedding dimensions must match")
    if neural_embeddings.shape[0] != len(labels):
        raise ValueError("len(labels) must equal number of neural samples")
    if text_embeddings.shape[0] != len(vocab):
        raise ValueError("len(vocab) must equal number of text embeddings")
    missing = sorted(set(labels) - set(vocab))
    if missing:
        raise ValueError(f"Labels missing from vocab: {missing[:10]}")
    

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
    _validate_retrieval_inputs(neural_embeddings, text_embeddings, labels, vocab)
    
    # Normalize embeddings for cosine similarity via dot product
    neural_norm = neural_embeddings / (np.linalg.norm(neural_embeddings, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix: (n_samples, vocab_size)
    similarity = neural_norm @ text_norm.T

    # Descending rank indices for each sample
    ranked_indices = np.argsort(-similarity, axis=1)

    vocab_index = {word: i for i, word in enumerate(vocab)}

    reciprocal_ranks = []
    hits = {k: 0 for k in k_values}

    for i, label in enumerate(labels):
        gt_idx = vocab_index.get(label)
        if gt_idx is None:
            continue

        ranked = ranked_indices[i]
        # np.where returns a tuple; [0][0] gives the position
        rank_positions = np.where(ranked == gt_idx)[0]
        if len(rank_positions) == 0:
            reciprocal_ranks.append(0.0)
            continue

        rank = rank_positions[0] + 1  # 1-indexed
        reciprocal_ranks.append(1.0 / rank)

        for k in k_values:
            if rank <= k:
                hits[k] += 1

    n = len(labels)
    metrics = {f"top{k}": hits[k] / n for k in k_values}
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return metrics


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
    _validate_retrieval_inputs(neural_embeddings, text_embeddings, labels, vocab)
    
    # Group sample indices by subject
    subject_map = defaultdict(list)
    for i, sid in enumerate(subject_ids):
        subject_map[sid].append(i)

    per_subject = {}
    for sid, indices in subject_map.items():
        s_neural = neural_embeddings[np.array(indices)]
        s_labels = [labels[i] for i in indices]
        per_subject[sid] = compute_retrieval_metrics(
            s_neural, text_embeddings, s_labels, vocab, k_values
        )

    # Mean and std across subjects (not pooled samples)
    subject_metrics = list(per_subject.values())
    all_keys = list(subject_metrics[0].keys())

    mean_metrics, std_metrics = {}, {}
    for key in all_keys:
        vals = np.array([m[key] for m in subject_metrics])
        mean_metrics[key] = float(vals.mean())
        std_metrics[key] = float(vals.std())

    return {
        "per_subject": per_subject,
        "mean": mean_metrics,
        "std": std_metrics,
    }


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
    #converts every embedding to unit length 
    def normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / norms

    eeg_norm = normalize(eeg_embeddings)
    meg_norm = normalize(meg_embeddings)

    meg_label_map = defaultdict(list)
    for i, label in enumerate(meg_labels):
        meg_label_map[label].append(i)

    # one EEG sample and one MEG sample that share the same word label
    matched_eeg_idx, matched_meg_idx = [], []
    for i, label in enumerate(eeg_labels):
        if label in meg_label_map:
            matched_eeg_idx.append(i)
            matched_meg_idx.append(meg_label_map[label][0])

    if not matched_eeg_idx:
        return {"matched_similarity": 0.0, "random_similarity": 0.0,
                "alignment_gap": 0.0, "n_matched_pairs": 0}

    #mean cosine similarity across all matched pairs
    matched_eeg = eeg_norm[matched_eeg_idx]
    matched_meg = meg_norm[matched_meg_idx]
    matched_similarity = float(np.sum(matched_eeg * matched_meg, axis=1).mean())

    #random: null baseline — you shuffle the MEG indices so each EEG embedding is paired with a random, unrelated MEG embedding
    #Alignment gap is the number you actually report. Positive means the model learned something real. A gap near zero even with decent matched similarity means your random baseline is inflated
    rng = np.random.default_rng(seed=42)
    shuffled_meg_idx = rng.permutation(matched_meg_idx)
    random_similarity = float(np.sum(matched_eeg * meg_norm[shuffled_meg_idx], axis=1).mean())

    return {
        "matched_similarity": matched_similarity,
        "random_similarity": random_similarity,
        "alignment_gap": matched_similarity - random_similarity,
        "n_matched_pairs": len(matched_eeg_idx),
    }


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
    _validate_retrieval_inputs(neural_embeddings, text_embeddings, labels, vocab)
    
    if confidence_thresholds is None:
        confidence_thresholds = np.linspace(0, 1, 50)

    # Normalize and compute full similarity matrix (n_samples, vocab_size)
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    sim = normalize(neural_embeddings) @ normalize(text_embeddings).T

    # Sort descending; confidence = gap between top-1 and top-2 scores
    sorted_sims = np.sort(sim, axis=1)[:, ::-1]
    confidence_scores = sorted_sims[:, 0] - sorted_sims[:, 1]

    # Ground truth correctness per sample
    # precomputes which samples the model got right before any abstention logic
    vocab_index = {w: i for i, w in enumerate(vocab)}
    gt_indices = np.array([vocab_index[l] for l in labels])
    predicted = np.argmax(sim, axis=1)
    correct = predicted == gt_indices


    # each threshold, any sample whose confidence score falls below it is "abstained" — the model refuses to answer
    coverage, accuracy = [], []
    for t in confidence_thresholds:
        retained = confidence_scores >= t
        n_retained = retained.sum()
        coverage.append(n_retained / len(labels))
        accuracy.append(correct[retained].mean() if n_retained > 0 else 0.0)

    coverage = np.array(coverage)
    accuracy = np.array(accuracy)

    # Find highest-coverage threshold that still hits 80% accuracy
    # operating point where the model hits your accuracy target while abstaining as little as possible
    above_80 = np.where(accuracy >= 0.8)[0]
    best_threshold_80pct = (
        float(confidence_thresholds[above_80[np.argmax(coverage[above_80])]])
        if len(above_80) > 0 else None
    )

    return {
        "thresholds": confidence_thresholds,
        "coverage":   coverage,
        "accuracy":   accuracy,
        "best_threshold_80pct": best_threshold_80pct,
    }


def extract_failure_cases(
    neural_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list,
    vocab: list,
    max_cases: int = 10,
) -> list:
    """
    Return representative failure cases:
    [
        {
            "true_label": "apple",
            "predicted_label": "banana",
            "top3_predictions": ["banana", "pear", "apple"],
            "scores": [0.81, 0.73, 0.69],
            "rank": 3,
        },
        ...
    ]
    """
    

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
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []

    # Helper: appends a line to the report buffer
    def w(line=""):
        lines.append(line)

    # ------------------------------------------------------------------ #
    # Section 1: Per-modality retrieval metrics
    # test_metrics is expected to be a dict of dicts, e.g.:
    #   {"eeg": {"top1": 0.25, "top5": 0.6, ...}, "meg": {...}}
    # ------------------------------------------------------------------ #
    w("# Stage 1 Evaluation Report")
    w()
    w("## 1. Per-modality retrieval metrics")
    w()
    w("| Modality | Top-1 | Top-5 | Top-10 | MRR |")
    w("|----------|-------|-------|--------|-----|")
    for modality, metrics in test_metrics.items():
        w(f"| {modality.upper()} "
          f"| {metrics['top1']:.3f} "
          f"| {metrics['top5']:.3f} "
          f"| {metrics['top10']:.3f} "
          f"| {metrics['mrr']:.3f} |")

    # ------------------------------------------------------------------ #
    # Section 2: Cross-subject generalization
    # per_subject is a dict of {subject_id: metrics_dict}
    # mean/std are dicts of {metric_name: float}, averaged across subjects
    # ------------------------------------------------------------------ #
    w()
    w("## 2. Cross-subject generalization")
    w()
    w("### Per-subject breakdown")
    w()
    w("| Subject | Top-1 | Top-5 | Top-10 | MRR |")
    w("|---------|-------|-------|--------|-----|")
    for subject, metrics in cross_subject_metrics["per_subject"].items():
        w(f"| {subject} "
          f"| {metrics['top1']:.3f} "
          f"| {metrics['top5']:.3f} "
          f"| {metrics['top10']:.3f} "
          f"| {metrics['mrr']:.3f} |")

    w()
    w("### Mean ± std across subjects")
    w()
    mean = cross_subject_metrics["mean"]
    std  = cross_subject_metrics["std"]
    w("| Metric | Mean | Std |")
    w("|--------|------|-----|")
    for key in ["top1", "top5", "top10", "mrr"]:
        w(f"| {key} | {mean[key]:.3f} | {std[key]:.3f} |")

    # ------------------------------------------------------------------ #
    # Section 3: Cross-modal alignment
    # matched_similarity: how similar same-word EEG/MEG pairs are
    # random_similarity:  baseline — unrelated pairs
    # alignment_gap:      matched - random; must be positive for real alignment
    # ------------------------------------------------------------------ #
    w()
    w("## 3. Cross-modal alignment")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Matched similarity  | {alignment_metrics['matched_similarity']:.3f} |")
    w(f"| Random similarity   | {alignment_metrics['random_similarity']:.3f} |")
    w(f"| Alignment gap       | {alignment_metrics['alignment_gap']:.3f} |")
    w(f"| Matched pairs found | {alignment_metrics['n_matched_pairs']} |")

    # Flag if the gap is negative — this would mean random pairs are closer
    # on average than matched pairs, which indicates the shared space is broken
    if alignment_metrics["alignment_gap"] <= 0:
        w()
        w("> **Warning:** alignment gap is non-positive. "
          "The shared embedding space may not be meaningfully aligned.")

    # ------------------------------------------------------------------ #
    # Section 4: Abstention curve
    # We sample 10 evenly spaced points from the threshold sweep to keep
    # the table readable — logging every threshold would produce 50 rows
    # ------------------------------------------------------------------ #
    w()
    w("## 4. Abstention curve")
    w()
    thresholds = abstention_metrics["thresholds"]
    coverage   = abstention_metrics["coverage"]
    accuracy   = abstention_metrics["accuracy"]
    best_t     = abstention_metrics["best_threshold_80pct"]

    w("| Threshold | Coverage | Accuracy |")
    w("|-----------|----------|----------|")
    n = len(thresholds)
    sample_indices = [int(i * (n - 1) / 9) for i in range(10)]  # 10 evenly spaced rows
    for i in sample_indices:
        w(f"| {thresholds[i]:.3f} | {coverage[i]:.3f} | {accuracy[i]:.3f} |")

    w()
    if best_t is not None:
        w(f"**Best threshold for ≥80% accuracy with maximum coverage:** `{best_t:.3f}`")
    else:
        w("**Note:** 80% accuracy was not achievable at any threshold.")

    # ------------------------------------------------------------------ #
    # Section 5: Failure cases
    # Each failure case is a dict with keys:
    #   'true_label'      — the correct word
    #   'top3_predictions'— list of the model's top 3 guesses
    #   'scores'          — corresponding cosine similarity scores
    # We cap at 10 cases regardless of how many were passed in
    # ------------------------------------------------------------------ #
    w()
    w("## 5. Representative failure cases")
    w()
    w("| # | True label | Top-3 predictions | Scores |")
    w("|---|------------|-------------------|--------|")
    for idx, case in enumerate(failure_cases[:10], 1):
        preds  = ", ".join(case["top3_predictions"])
        scores = ", ".join(f"{s:.3f}" for s in case["scores"])
        w(f"| {idx} | {case['true_label']} | {preds} | {scores} |")

    # ------------------------------------------------------------------ #
    # Write the accumulated lines to disk as a single markdown file
    # ------------------------------------------------------------------ #
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    return output_path
    
