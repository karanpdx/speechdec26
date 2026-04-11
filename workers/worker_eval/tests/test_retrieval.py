"""
Tests for Stage 1 retrieval evaluation metrics.

All tests use synthetic embeddings from stubs/generate_synthetic.py.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stubs.generate_synthetic import make_retrieval_inputs


class TestComputeRetrievalMetrics:
    def test_returns_required_keys(self):
        from src.evaluation.retrieval import compute_retrieval_metrics
        d = make_retrieval_inputs(n_samples=50, vocab_size=20)
        result = compute_retrieval_metrics(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"]
        )
        assert "top1" in result
        assert "top5" in result
        assert "top10" in result
        assert "mrr" in result

    def test_perfect_retrieval_gives_top1_1(self):
        """Neural embeddings identical to their correct text embeddings → top1 = 1.0"""
        from src.evaluation.retrieval import compute_retrieval_metrics
        vocab = [f"word_{i}" for i in range(10)]
        text_embs = np.random.randn(10, 64).astype(np.float32)
        text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)
        label_indices = np.arange(10)
        labels = [vocab[i] for i in label_indices]
        # Neural = text (perfect)
        neural_embs = text_embs[label_indices]
        result = compute_retrieval_metrics(neural_embs, text_embs, labels, vocab)
        assert result["top1"] == pytest.approx(1.0, abs=1e-5)

    def test_random_retrieval_near_chance(self):
        """Random embeddings should give top1 ≈ 1/vocab_size."""
        from src.evaluation.retrieval import compute_retrieval_metrics
        np.random.seed(0)
        vocab_size = 100
        n_samples = 500
        vocab = [f"word_{i}" for i in range(vocab_size)]
        text_embs = np.random.randn(vocab_size, 64).astype(np.float32)
        text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)
        neural_embs = np.random.randn(n_samples, 64).astype(np.float32)
        neural_embs /= np.linalg.norm(neural_embs, axis=1, keepdims=True)
        labels = [vocab[i % vocab_size] for i in range(n_samples)]
        result = compute_retrieval_metrics(neural_embs, text_embs, labels, vocab)
        # top1 should be near 1/100 = 0.01 (allow generous range)
        assert result["top1"] < 0.10

    def test_metrics_are_in_0_1(self):
        from src.evaluation.retrieval import compute_retrieval_metrics
        d = make_retrieval_inputs(n_samples=100, vocab_size=30)
        result = compute_retrieval_metrics(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"]
        )
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0, 1]"

    def test_mrr_lower_bounded_by_top1(self):
        """MRR >= top1 (rank 1 contributes 1.0 to MRR, higher ranks contribute less)."""
        from src.evaluation.retrieval import compute_retrieval_metrics
        d = make_retrieval_inputs(n_samples=100, vocab_size=20, top1_accuracy=0.3)
        result = compute_retrieval_metrics(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"]
        )
        assert result["mrr"] >= result["top1"] - 1e-5

    def test_top1_leq_top5_leq_top10(self):
        from src.evaluation.retrieval import compute_retrieval_metrics
        d = make_retrieval_inputs(n_samples=200, vocab_size=50)
        result = compute_retrieval_metrics(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"]
        )
        assert result["top1"] <= result["top5"] + 1e-5
        assert result["top5"] <= result["top10"] + 1e-5

    def test_custom_k_values(self):
        from src.evaluation.retrieval import compute_retrieval_metrics
        d = make_retrieval_inputs(n_samples=50, vocab_size=20)
        result = compute_retrieval_metrics(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"],
            k_values=[1, 3]
        )
        assert "top1" in result
        assert "top3" in result
        assert "top5" not in result


class TestCrossSubjectGeneralization:
    def test_returns_per_subject_and_mean(self):
        from src.evaluation.retrieval import compute_cross_subject_generalization
        d = make_retrieval_inputs(n_samples=100, vocab_size=20, n_subjects=4)
        result = compute_cross_subject_generalization(
            d["neural_embeddings"], d["text_embeddings"],
            d["labels"], d["vocab"], d["subject_ids"]
        )
        assert "per_subject" in result
        assert "mean" in result
        assert "std" in result

    def test_per_subject_keys_match_subject_ids(self):
        from src.evaluation.retrieval import compute_cross_subject_generalization
        d = make_retrieval_inputs(n_samples=80, vocab_size=20, n_subjects=4)
        result = compute_cross_subject_generalization(
            d["neural_embeddings"], d["text_embeddings"],
            d["labels"], d["vocab"], d["subject_ids"]
        )
        expected_subjects = set(d["subject_ids"])
        assert set(result["per_subject"].keys()) == expected_subjects

    def test_mean_is_mean_of_per_subject_not_pooled(self):
        """Mean should be computed over subjects, not over all samples."""
        from src.evaluation.retrieval import compute_cross_subject_generalization
        d = make_retrieval_inputs(n_samples=100, vocab_size=20, n_subjects=3)
        result = compute_cross_subject_generalization(
            d["neural_embeddings"], d["text_embeddings"],
            d["labels"], d["vocab"], d["subject_ids"]
        )
        # Mean across 3 subjects (not 100 samples)
        per_subj_top1 = [v["top1"] for v in result["per_subject"].values()]
        expected_mean = np.mean(per_subj_top1)
        assert result["mean"]["top1"] == pytest.approx(expected_mean, abs=1e-5)


class TestCrossModalAlignment:
    def test_returns_required_keys(self):
        from src.evaluation.retrieval import compute_cross_modal_alignment
        vocab = [f"word_{i}" for i in range(10)]
        eeg_embs = np.random.randn(50, 64).astype(np.float32)
        meg_embs = np.random.randn(50, 64).astype(np.float32)
        eeg_labels = [vocab[i % 10] for i in range(50)]
        meg_labels = [vocab[i % 10] for i in range(50)]
        result = compute_cross_modal_alignment(eeg_embs, meg_embs, eeg_labels, meg_labels)
        assert "matched_similarity" in result
        assert "random_similarity" in result
        assert "alignment_gap" in result
        assert "n_matched_pairs" in result

    def test_identical_embeddings_give_gap_of_1(self):
        """If EEG and MEG embeddings are identical for matched words, gap should be large."""
        from src.evaluation.retrieval import compute_cross_modal_alignment
        embs = np.random.randn(20, 64).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        labels = [f"word_{i}" for i in range(20)]
        result = compute_cross_modal_alignment(embs, embs, labels, labels)
        assert result["alignment_gap"] > 0.5

    def test_random_embeddings_give_near_zero_gap(self):
        from src.evaluation.retrieval import compute_cross_modal_alignment
        np.random.seed(1)
        eeg = np.random.randn(100, 64).astype(np.float32)
        meg = np.random.randn(100, 64).astype(np.float32)
        labels = [f"word_{i % 20}" for i in range(100)]
        result = compute_cross_modal_alignment(eeg, meg, labels, labels)
        assert abs(result["alignment_gap"]) < 0.2


class TestAbstentionCurve:
    def test_returns_required_keys(self):
        from src.evaluation.retrieval import compute_abstention_curve
        d = make_retrieval_inputs(n_samples=100, vocab_size=20)
        result = compute_abstention_curve(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"]
        )
        assert "thresholds" in result
        assert "coverage" in result
        assert "accuracy" in result
        assert "best_threshold_80pct" in result

    def test_coverage_decreases_with_threshold(self):
        """As threshold increases, coverage should be non-increasing."""
        from src.evaluation.retrieval import compute_abstention_curve
        d = make_retrieval_inputs(n_samples=200, vocab_size=30)
        result = compute_abstention_curve(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"]
        )
        coverage = result["coverage"]
        for i in range(len(coverage) - 1):
            assert coverage[i] >= coverage[i + 1] - 1e-5

    def test_zero_threshold_covers_all(self):
        """At threshold=0, no samples are abstained."""
        from src.evaluation.retrieval import compute_abstention_curve
        d = make_retrieval_inputs(n_samples=50, vocab_size=10)
        thresholds = np.array([0.0])
        result = compute_abstention_curve(
            d["neural_embeddings"], d["text_embeddings"], d["labels"], d["vocab"],
            confidence_thresholds=thresholds
        )
        assert result["coverage"][0] == pytest.approx(1.0, abs=1e-5)
