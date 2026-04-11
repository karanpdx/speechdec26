"""
Tests for Stage 2 reconstruction evaluation metrics.

All tests use synthetic data from stubs/generate_synthetic.py.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stubs.generate_synthetic import make_reconstruction_inputs, make_round_trip_inputs


class TestComputeN400Correlation:
    def test_returns_required_keys(self):
        from src.evaluation.reconstruction import compute_n400_correlation
        d = make_reconstruction_inputs(n_samples=40, n_channels=64, n_timepoints=175)
        labels = [f"word_{i % 10}" for i in range(40)]
        result = compute_n400_correlation(
            d["pred_eeg"], d["real_eeg"], labels, d["ch_names"], d["sfreq"]
        )
        assert "pearson_r" in result
        assert "p_value" in result
        assert "n_words" in result
        assert "pred_n400" in result
        assert "real_n400" in result

    def test_correlated_signals_give_positive_r(self):
        """Predicted signals with r_erp=0.8 should give clearly positive Pearson r."""
        from src.evaluation.reconstruction import compute_n400_correlation
        d = make_reconstruction_inputs(n_samples=60, r_erp=0.8, seed=0)
        labels = [f"word_{i % 15}" for i in range(60)]
        result = compute_n400_correlation(
            d["pred_eeg"], d["real_eeg"], labels, d["ch_names"], d["sfreq"]
        )
        assert result["pearson_r"] > 0

    def test_random_signals_give_near_zero_r(self):
        """Uncorrelated predicted signals should give Pearson r near 0."""
        from src.evaluation.reconstruction import compute_n400_correlation
        d = make_reconstruction_inputs(n_samples=100, r_erp=0.0, seed=42)
        labels = [f"word_{i % 20}" for i in range(100)]
        result = compute_n400_correlation(
            d["pred_eeg"], d["real_eeg"], labels, d["ch_names"], d["sfreq"]
        )
        assert abs(result["pearson_r"]) < 0.5


class TestComputeFmriSpatialCorrelation:
    def test_returns_required_keys(self):
        from src.evaluation.reconstruction import compute_fmri_spatial_correlation
        d = make_reconstruction_inputs(n_samples=20, n_voxels=500)
        result = compute_fmri_spatial_correlation(d["pred_fmri"], d["real_fmri"])
        assert "per_word_r" in result
        assert "mean_r" in result
        assert "std_r" in result

    def test_perfect_reconstruction_gives_r_1(self):
        from src.evaluation.reconstruction import compute_fmri_spatial_correlation
        betas = np.random.randn(10, 200).astype(np.float32)
        result = compute_fmri_spatial_correlation(betas, betas)
        assert result["mean_r"] == pytest.approx(1.0, abs=1e-4)

    def test_per_word_shape(self):
        from src.evaluation.reconstruction import compute_fmri_spatial_correlation
        d = make_reconstruction_inputs(n_samples=15, n_voxels=300)
        result = compute_fmri_spatial_correlation(d["pred_fmri"], d["real_fmri"])
        assert len(result["per_word_r"]) == 15

    def test_mean_is_mean_of_per_word(self):
        from src.evaluation.reconstruction import compute_fmri_spatial_correlation
        d = make_reconstruction_inputs(n_samples=10, n_voxels=200)
        result = compute_fmri_spatial_correlation(d["pred_fmri"], d["real_fmri"])
        assert result["mean_r"] == pytest.approx(np.mean(result["per_word_r"]), abs=1e-5)


class TestComputeRoundTripSimilarity:
    def test_returns_required_keys(self):
        from src.evaluation.reconstruction import compute_round_trip_similarity
        d = make_round_trip_inputs(n_words=20)
        result = compute_round_trip_similarity(
            d["original_embeddings"], d["round_trip_embeddings"]
        )
        assert "per_word_similarity" in result
        assert "mean" in result
        assert "std" in result
        assert "fraction_above_0.5" in result
        assert "fraction_above_0.7" in result

    def test_identical_embeddings_give_similarity_1(self):
        from src.evaluation.reconstruction import compute_round_trip_similarity
        embs = np.random.randn(10, 64).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        result = compute_round_trip_similarity(embs, embs)
        assert result["mean"] == pytest.approx(1.0, abs=1e-4)

    def test_high_similarity_stub(self):
        """Synthetic data with r=0.7 should give mean near 0.7."""
        from src.evaluation.reconstruction import compute_round_trip_similarity
        d = make_round_trip_inputs(n_words=100, round_trip_similarity=0.7)
        result = compute_round_trip_similarity(
            d["original_embeddings"], d["round_trip_embeddings"]
        )
        assert result["mean"] == pytest.approx(0.7, abs=0.1)

    def test_fraction_above_threshold(self):
        from src.evaluation.reconstruction import compute_round_trip_similarity
        embs = np.random.randn(20, 32).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        result = compute_round_trip_similarity(embs, embs)
        assert result["fraction_above_0.5"] == pytest.approx(1.0, abs=1e-5)
        assert result["fraction_above_0.7"] == pytest.approx(1.0, abs=1e-5)

    def test_per_word_shape(self):
        from src.evaluation.reconstruction import compute_round_trip_similarity
        d = make_round_trip_inputs(n_words=25)
        result = compute_round_trip_similarity(
            d["original_embeddings"], d["round_trip_embeddings"]
        )
        assert len(result["per_word_similarity"]) == 25


class TestComputeErp:
    def test_grand_average_shape(self):
        from src.evaluation.reconstruction import compute_erp
        epochs = np.random.randn(50, 64, 175).astype(np.float32)
        erp = compute_erp(epochs, sfreq=256.0)
        assert erp.shape == (64, 175)

    def test_grand_average_is_mean_over_epochs(self):
        from src.evaluation.reconstruction import compute_erp
        epochs = np.random.randn(10, 8, 50).astype(np.float32)
        erp = compute_erp(epochs, sfreq=256.0)
        expected = epochs.mean(axis=0)
        np.testing.assert_allclose(erp, expected, atol=1e-5)
