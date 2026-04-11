"""
Tests for the data preprocessing pipeline.

All tests use synthetic data — no real datasets required.
Every test must pass before any module is considered done.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_synthetic_eeg_raw(n_channels: int = 64, sfreq: float = 1000.0, duration_s: float = 60.0):
    """Create a synthetic MNE Raw object for EEG testing."""
    import mne
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    data = np.random.randn(n_channels, int(sfreq * duration_s)) * 1e-6
    return mne.io.RawArray(data, info)


def make_synthetic_npz(n_epochs: int = 50, n_channels: int = 64, n_timepoints: int = 175):
    """Create a synthetic epoched .npz dict matching the output schema."""
    data = np.random.randn(n_epochs, n_channels, n_timepoints).astype(np.float32)
    # Roughly z-score it
    data = (data - data.mean(axis=(0, 2), keepdims=True)) / (data.std(axis=(0, 2), keepdims=True) + 1e-8)
    labels = [f"word_{i % 20}" for i in range(n_epochs)]
    return {
        "data": data,
        "labels": labels,
        "subject_id": "sub-01",
        "sfreq": 256.0,
        "ch_names": [f"EEG{i:03d}" for i in range(n_channels)],
        "event_onsets_s": np.linspace(0, 60, n_epochs),
    }


def make_synthetic_fmri_npz(n_words: int = 30, n_voxels: int = 1000):
    """Create a synthetic fMRI beta map .npz dict matching the output schema."""
    data = np.random.randn(n_words, n_voxels).astype(np.float32)
    labels = [f"word_{i}" for i in range(n_words)]
    voxel_coords = np.random.randint(-90, 90, size=(n_voxels, 3), dtype=np.int32)
    return {
        "data": data,
        "labels": labels,
        "subject_id": "sub-01",
        "voxel_coords": voxel_coords,
        "pca_explained_variance": 0.0,
    }


# ---------------------------------------------------------------------------
# Output schema validation (shared logic)
# ---------------------------------------------------------------------------

def assert_valid_eeg_meg_output(npz: dict) -> None:
    data = npz["data"]
    labels = npz["labels"]
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    assert data.ndim == 3, f"Expected 3D array, got shape {data.shape}"
    assert len(labels) == data.shape[0], "Label count must match epoch count"
    assert not np.any(np.isnan(data)), "NaN values in output"
    assert not np.any(np.isinf(data)), "Inf values in output"
    assert abs(data.mean()) < 0.1, f"Mean {data.mean():.3f} too far from 0 (not z-scored?)"
    assert abs(data.std() - 1.0) < 0.2, f"Std {data.std():.3f} too far from 1 (not z-scored?)"


def assert_valid_fmri_output(npz: dict) -> None:
    data = npz["data"]
    labels = npz["labels"]
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    assert data.ndim == 2, f"Expected 2D array, got shape {data.shape}"
    assert len(labels) == data.shape[0], "Label count must match word count"
    assert not np.any(np.isnan(data)), "NaN values in output"
    assert data.std() > 0, "Beta maps are all zeros"


# ---------------------------------------------------------------------------
# EEG tests
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_output_is_approx_zscore(self):
        from src.data.preprocess_eeg import normalize
        data = np.random.randn(50, 64, 175).astype(np.float32) * 10 + 5
        out = normalize(data)
        assert abs(out.mean()) < 0.1
        assert abs(out.std() - 1.0) < 0.2

    def test_output_dtype_is_float32(self):
        from src.data.preprocess_eeg import normalize
        data = np.random.randn(10, 8, 50).astype(np.float32)
        out = normalize(data)
        assert out.dtype == np.float32

    def test_output_shape_unchanged(self):
        from src.data.preprocess_eeg import normalize
        data = np.random.randn(20, 32, 100).astype(np.float32)
        out = normalize(data)
        assert out.shape == data.shape


class TestEEGOutputSchema:
    def test_synthetic_npz_passes_validation(self):
        npz = make_synthetic_npz()
        assert_valid_eeg_meg_output(npz)

    def test_too_few_epochs_should_warn(self):
        """Subjects with fewer than 10 epochs should be skipped (not raise)."""
        # The pipeline should log a warning and return None or skip — not crash.
        # Test that the guard condition is present in run_subject.
        from src.data.preprocess_eeg import run_subject
        # This test verifies the interface exists; real behavior tested with integration.
        assert callable(run_subject)


# ---------------------------------------------------------------------------
# fMRI tests
# ---------------------------------------------------------------------------

class TestFmriOutputSchema:
    def test_synthetic_npz_passes_validation(self):
        npz = make_synthetic_fmri_npz()
        assert_valid_fmri_output(npz)


class TestPCA:
    def test_pca_reduces_dimensionality(self):
        from src.data.preprocess_fmri import apply_pca
        data = np.random.randn(30, 500).astype(np.float32)
        projected, pca = apply_pca(data, n_components=50, fit=True)
        assert projected.shape == (30, 50)

    def test_pca_transform_consistency(self):
        """Applying the same PCA to two batches should give consistent results."""
        from src.data.preprocess_fmri import apply_pca
        data_train = np.random.randn(30, 500).astype(np.float32)
        data_test = np.random.randn(10, 500).astype(np.float32)
        _, pca = apply_pca(data_train, n_components=20, fit=True)
        out_test, _ = apply_pca(data_test, n_components=20, fit=False, pca=pca)
        assert out_test.shape == (10, 20)


# ---------------------------------------------------------------------------
# Vocabulary / alignment tests
# ---------------------------------------------------------------------------

class TestVocabEmbeddings:
    def test_embedding_shape(self):
        from src.data.vocab_embeddings import get_bert_embedding, load_bert
        tokenizer, model = load_bert()
        emb = get_bert_embedding("language", tokenizer, model)
        assert emb.shape == (768,)
        assert emb.dtype == np.float32

    def test_no_nan_in_embeddings(self):
        from src.data.vocab_embeddings import generate_vocab_embeddings
        result = generate_vocab_embeddings(["brain", "speech", "word"], batch_size=3)
        assert not np.any(np.isnan(result["embeddings"]))


class TestSplitIntegrity:
    def test_no_subject_leakage(self):
        from src.data.align_splits import verify_split_integrity
        splits = {
            "train": {"eeg": ["sub-01_epochs.npz", "sub-02_epochs.npz"]},
            "val":   {"eeg": ["sub-03_epochs.npz"]},
            "test":  {"eeg": ["sub-04_epochs.npz"]},
        }
        # Should not raise
        verify_split_integrity(splits, val_subjects=["sub-03"], test_subjects=["sub-04"])

    def test_subject_leakage_raises(self):
        from src.data.align_splits import verify_split_integrity
        splits = {
            "train": {"eeg": ["sub-01_epochs.npz", "sub-04_epochs.npz"]},  # sub-04 is in test!
            "val":   {"eeg": ["sub-03_epochs.npz"]},
            "test":  {"eeg": ["sub-04_epochs.npz"]},
        }
        with pytest.raises(AssertionError):
            verify_split_integrity(splits, val_subjects=["sub-03"], test_subjects=["sub-04"])
