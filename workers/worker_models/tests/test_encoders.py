"""
Tests for encoder architectures.

All tests use synthetic random tensors — no data or training required.
These are the exact tests specified in AGENTS.md plus additional edge cases.
"""

import pytest
import torch
import torch.nn.functional as F


class TestEEGEncoder:
    def test_output_shape(self):
        from src.models.encoders import EEGEncoder
        enc = EEGEncoder(n_channels=64, n_timepoints=175, embed_dim=768)
        x = torch.randn(8, 64, 175)
        out = enc(x)
        assert out.shape == (8, 768)

    def test_output_shape_custom_embed_dim(self):
        from src.models.encoders import EEGEncoder
        enc = EEGEncoder(n_channels=32, n_timepoints=128, embed_dim=256)
        x = torch.randn(4, 32, 128)
        out = enc(x)
        assert out.shape == (4, 256)

    def test_batch_size_1(self):
        from src.models.encoders import EEGEncoder
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(1, 64, 175)
        out = enc(x)
        assert out.shape == (1, 768)

    def test_output_is_finite(self):
        from src.models.encoders import EEGEncoder
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(8, 64, 175)
        out = enc(x)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        from src.models.encoders import EEGEncoder
        enc = EEGEncoder(n_channels=64, n_timepoints=175)
        x = torch.randn(4, 64, 175, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


class TestMEGEncoder:
    def test_output_shape(self):
        from src.models.encoders import MEGEncoder
        enc = MEGEncoder(n_channels=306, n_timepoints=175, embed_dim=768)
        x = torch.randn(8, 306, 175)
        out = enc(x)
        assert out.shape == (8, 768)

    def test_different_channel_count(self):
        from src.models.encoders import MEGEncoder
        enc = MEGEncoder(n_channels=102, n_timepoints=175)
        x = torch.randn(4, 102, 175)
        out = enc(x)
        assert out.shape == (4, 768)


class TestTemporalWeightSharing:
    def test_weights_are_same_object(self):
        from src.models.encoders import EEGEncoder, MEGEncoder
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=True)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
        eeg_enc.share_temporal_weights(meg_enc)
        # The underlying parameter data_ptr must be identical
        assert eeg_enc.temporal_conv.weight.data_ptr() == meg_enc.temporal_conv.weight.data_ptr()

    def test_modifying_one_affects_other(self):
        from src.models.encoders import EEGEncoder, MEGEncoder
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=True)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
        eeg_enc.share_temporal_weights(meg_enc)
        # Mutate eeg weights in-place, check meg sees the change
        with torch.no_grad():
            eeg_enc.temporal_conv.weight.fill_(0.0)
        assert (meg_enc.temporal_conv.weight == 0.0).all()

    def test_share_requires_flag(self):
        from src.models.encoders import EEGEncoder, MEGEncoder
        eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=False)
        meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
        with pytest.raises(RuntimeError):
            eeg_enc.share_temporal_weights(meg_enc)


class TestFmriEncoder:
    def test_output_shape(self):
        from src.models.encoders import fMRIEncoder
        enc = fMRIEncoder(n_voxels=1000, embed_dim=768)
        x = torch.randn(8, 1000)
        out = enc(x)
        assert out.shape == (8, 768)

    def test_small_voxel_count(self):
        from src.models.encoders import fMRIEncoder
        enc = fMRIEncoder(n_voxels=50, embed_dim=256)
        x = torch.randn(4, 50)
        out = enc(x)
        assert out.shape == (4, 256)

    def test_large_voxel_count_warns(self):
        """n_voxels > 10000 should log a warning (not raise)."""
        import logging
        from src.models.encoders import fMRIEncoder
        with pytest.warns(None):  # Should not raise; warning goes to logger
            enc = fMRIEncoder(n_voxels=15000, embed_dim=768)
        x = torch.randn(2, 15000)
        out = enc(x)
        assert out.shape == (2, 768)

    def test_gradients_flow(self):
        from src.models.encoders import fMRIEncoder
        enc = fMRIEncoder(n_voxels=500)
        x = torch.randn(4, 500, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


class TestSharedEmbeddingProjector:
    def test_output_shape(self):
        from src.models.encoders import SharedEmbeddingProjector
        proj = SharedEmbeddingProjector(bert_dim=768, embed_dim=768)
        x = torch.randn(8, 768)
        out = proj(x)
        assert out.shape == (8, 768)

    def test_different_embed_dim(self):
        from src.models.encoders import SharedEmbeddingProjector
        proj = SharedEmbeddingProjector(bert_dim=768, embed_dim=512)
        x = torch.randn(4, 768)
        out = proj(x)
        assert out.shape == (4, 512)


class TestSubjectEmbedding:
    def test_output_shape(self):
        from src.models.encoders import SubjectEmbedding
        emb = SubjectEmbedding(n_subjects=20, subject_embed_dim=64)
        ids = torch.randint(0, 20, (8,))
        out = emb(ids)
        assert out.shape == (8, 64)

    def test_get_mean_embedding_shape(self):
        from src.models.encoders import SubjectEmbedding
        emb = SubjectEmbedding(n_subjects=10, subject_embed_dim=64)
        mean = emb.get_mean_embedding()
        assert mean.shape == (1, 64)

    def test_get_mean_embedding_is_mean_of_all(self):
        from src.models.encoders import SubjectEmbedding
        emb = SubjectEmbedding(n_subjects=4, subject_embed_dim=8)
        all_ids = torch.arange(4)
        all_embs = emb(all_ids)
        expected_mean = all_embs.mean(dim=0, keepdim=True)
        actual_mean = emb.get_mean_embedding()
        assert torch.allclose(actual_mean, expected_mean, atol=1e-5)
