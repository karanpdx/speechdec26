"""
Tests for encoder architectures in src/models/encoders.py.

Covers:
    - Output shapes for all encoders
    - Temporal weight sharing between EEGEncoder and MEGEncoder
    - Edge cases: large n_voxels warning, batch size 1, dtype preservation
    - Expected failures: wrong input shapes, wrong subject_id dtype
    - Debug-mode shape assertions
"""

import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.encoders import (
    EEGEncoder,
    MEGEncoder,
    SharedEmbeddingProjector,
    SubjectEmbedding,
    fMRIEncoder,
)

# ---------------------------------------------------------------------------
# EEGEncoder
# ---------------------------------------------------------------------------


class TestEEGEncoder:
    def test_output_shape_standard(self):
        """Happy path: standard EEG dimensions from AGENTS.md spec."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175, embed_dim=768)
        x = torch.randn(8, 64, 175)
        out = enc(x)
        assert out.shape == (8, 768), f"Expected (8, 768), got {out.shape}"

    def test_output_shape_small_batch(self):
        """Batch size of 1 should work."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175, embed_dim=768)
        x = torch.randn(1, 64, 175)
        out = enc(x)
        assert out.shape == (1, 768)

    def test_output_shape_custom_embed_dim(self):
        """Custom embed_dim is respected."""
        enc = EEGEncoder(n_channels=32, n_timepoints=256, embed_dim=512)
        x = torch.randn(4, 32, 256)
        out = enc(x)
        assert out.shape == (4, 512)

    def test_output_is_float32(self):
        """Output dtype should be float32."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(4, 64, 175)
        out = enc(x)
        assert out.dtype == torch.float32

    def test_output_no_nan(self):
        """Output must not contain NaN or Inf."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(8, 64, 175)
        out = enc(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_wrong_channel_count_raises(self):
        """Debug assertion fires when n_channels mismatch."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(4, 32, 175)   # wrong: 32 ≠ 64
        with pytest.raises((AssertionError, RuntimeError)):
            enc(x)

    def test_wrong_timepoints_raises(self):
        """Debug assertion fires when n_timepoints mismatch."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(4, 64, 200)   # wrong: 200 ≠ 175
        with pytest.raises((AssertionError, RuntimeError)):
            enc(x)

    def test_gradients_flow(self):
        """Gradients must flow through the encoder."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(4, 64, 175, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_parameters_exist(self):
        """Encoder must have trainable parameters."""
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        params = list(enc.parameters())
        assert len(params) > 0, "Encoder has no parameters"


# ---------------------------------------------------------------------------
# MEGEncoder
# ---------------------------------------------------------------------------


class TestMEGEncoder:
    def test_output_shape_standard(self):
        """Happy path: standard MEG dimensions from AGENTS.md spec."""
        enc = MEGEncoder(n_channels=306, n_timepoints=175, embed_dim=768)
        x = torch.randn(8, 306, 175)
        out = enc(x)
        assert out.shape == (8, 768), f"Expected (8, 768), got {out.shape}"

    def test_output_shape_custom(self):
        enc = MEGEncoder(n_channels=204, n_timepoints=200, embed_dim=512)
        x = torch.randn(4, 204, 200)
        out = enc(x)
        assert out.shape == (4, 512)

    def test_output_no_nan(self):
        enc = MEGEncoder(n_channels=306, n_timepoints=175)
        x = torch.randn(8, 306, 175)
        out = enc(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradients_flow(self):
        enc = MEGEncoder(n_channels=306, n_timepoints=175)
        x = torch.randn(4, 306, 175, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Temporal weight sharing
# ---------------------------------------------------------------------------


class TestTemporalWeightSharing:
    def test_shared_weights_same_data_ptr(self):
        """After sharing, both encoders must reference the same parameter object."""
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=True)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
        eeg_enc.share_temporal_weights(meg_enc)
        assert eeg_enc.temporal_conv.weight.data_ptr() == meg_enc.temporal_conv.weight.data_ptr(), (
            "After sharing, temporal_conv.weight must be the same parameter object"
        )

    def test_shared_weights_mutation_propagates(self):
        """Mutating EEG encoder weight must be visible in MEG encoder."""
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=True)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
        eeg_enc.share_temporal_weights(meg_enc)

        with torch.no_grad():
            eeg_enc.temporal_conv.weight.fill_(3.14)

        assert torch.allclose(
            meg_enc.temporal_conv.weight,
            torch.full_like(meg_enc.temporal_conv.weight, 3.14),
        ), "Weight mutation did not propagate to MEGEncoder"

    def test_share_temporal_without_flag_raises(self):
        """Calling share_temporal_weights when share_temporal=False must raise."""
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=False)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
        with pytest.raises(RuntimeError):
            eeg_enc.share_temporal_weights(meg_enc)

    def test_share_temporal_target_without_flag_raises(self):
        """Target MEGEncoder must also have share_temporal=True."""
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=True)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=False)
        with pytest.raises(RuntimeError):
            eeg_enc.share_temporal_weights(meg_enc)

    def test_encoders_still_produce_correct_shapes_after_sharing(self):
        """Shared weights must not break forward passes."""
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=True)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
        eeg_enc.share_temporal_weights(meg_enc)

        eeg_out = eeg_enc(torch.randn(4, 64, 175))
        meg_out = meg_enc(torch.randn(4, 306, 175))
        assert eeg_out.shape == (4, 768)
        assert meg_out.shape == (4, 768)


# ---------------------------------------------------------------------------
# fMRIEncoder
# ---------------------------------------------------------------------------


class TestFMRIEncoder:
    def test_output_shape_standard(self):
        """Happy path from AGENTS.md spec."""
        enc = fMRIEncoder(n_voxels=1000, embed_dim=768)
        x = torch.randn(8, 1000)
        out = enc(x)
        assert out.shape == (8, 768)

    def test_output_shape_small_voxels(self):
        enc = fMRIEncoder(n_voxels=128, embed_dim=512)
        x = torch.randn(4, 128)
        out = enc(x)
        assert out.shape == (4, 512)

    def test_large_voxels_logs_warning(self, caplog):
        """n_voxels > 10,000 must emit a warning."""
        with caplog.at_level(logging.WARNING, logger="src.models.encoders"):
            fMRIEncoder(n_voxels=15_000)
        assert any("10" in msg or "PCA" in msg for msg in caplog.messages), (
            "Expected a warning about large n_voxels / PCA recommendation"
        )

    def test_small_voxels_no_warning(self, caplog):
        """n_voxels ≤ 10,000 must NOT emit that warning."""
        with caplog.at_level(logging.WARNING, logger="src.models.encoders"):
            fMRIEncoder(n_voxels=1000)
        pca_warnings = [m for m in caplog.messages if "PCA" in m]
        assert len(pca_warnings) == 0

    def test_output_no_nan(self):
        enc = fMRIEncoder(n_voxels=1000)
        out = enc(torch.randn(8, 1000))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_wrong_voxel_count_raises(self):
        enc = fMRIEncoder(n_voxels=1000)
        with pytest.raises((AssertionError, RuntimeError)):
            enc(torch.randn(4, 500))  # wrong: 500 ≠ 1000

    def test_gradients_flow(self):
        enc = fMRIEncoder(n_voxels=1000)
        x = torch.randn(4, 1000, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SharedEmbeddingProjector
# ---------------------------------------------------------------------------


class TestSharedEmbeddingProjector:
    def test_output_shape_standard(self):
        proj = SharedEmbeddingProjector(bert_dim=768, embed_dim=768)
        x = torch.randn(8, 768)
        out = proj(x)
        assert out.shape == (8, 768)

    def test_output_shape_custom_dims(self):
        proj = SharedEmbeddingProjector(bert_dim=1024, embed_dim=512)
        x = torch.randn(4, 1024)
        out = proj(x)
        assert out.shape == (4, 512)

    def test_output_no_nan(self):
        proj = SharedEmbeddingProjector()
        out = proj(torch.randn(8, 768))
        assert not torch.isnan(out).any()

    def test_wrong_bert_dim_raises(self):
        proj = SharedEmbeddingProjector(bert_dim=768)
        with pytest.raises((AssertionError, RuntimeError)):
            proj(torch.randn(4, 512))

    def test_gradients_flow(self):
        proj = SharedEmbeddingProjector()
        x = torch.randn(4, 768, requires_grad=True)
        proj(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SubjectEmbedding
# ---------------------------------------------------------------------------


class TestSubjectEmbedding:
    def test_output_shape(self):
        emb = SubjectEmbedding(n_subjects=20, subject_embed_dim=64)
        ids = torch.randint(0, 20, (8,))
        out = emb(ids)
        assert out.shape == (8, 64)

    def test_output_shape_custom_dim(self):
        emb = SubjectEmbedding(n_subjects=10, subject_embed_dim=128)
        ids = torch.randint(0, 10, (4,))
        out = emb(ids)
        assert out.shape == (4, 128)

    def test_get_mean_embedding_shape(self):
        emb = SubjectEmbedding(n_subjects=20, subject_embed_dim=64)
        mean = emb.get_mean_embedding()
        assert mean.shape == (1, 64)

    def test_get_mean_embedding_is_mean(self):
        """get_mean_embedding must equal the mean of all subject embeddings."""
        emb = SubjectEmbedding(n_subjects=5, subject_embed_dim=16)
        expected = emb.embedding.weight.mean(dim=0, keepdim=True)
        actual = emb.get_mean_embedding()
        assert torch.allclose(actual, expected)

    def test_wrong_dtype_raises(self):
        """Float IDs should fail the dtype assertion."""
        emb = SubjectEmbedding(n_subjects=20, subject_embed_dim=64)
        ids = torch.rand(8)   # float, not long
        with pytest.raises((AssertionError, RuntimeError)):
            emb(ids)

    def test_single_subject(self):
        emb = SubjectEmbedding(n_subjects=1, subject_embed_dim=64)
        ids = torch.zeros(4, dtype=torch.long)
        out = emb(ids)
        assert out.shape == (4, 64)

    def test_different_subjects_different_embeddings(self):
        """Different subject IDs must produce different embeddings."""
        emb = SubjectEmbedding(n_subjects=10, subject_embed_dim=64)
        out_0 = emb(torch.tensor([0]))
        out_1 = emb(torch.tensor([1]))
        assert not torch.allclose(out_0, out_1)
