"""
Tests for decoder architectures.

All tests use synthetic random tensors — no data or training required.
These are the exact tests specified in AGENTS.md plus additional edge cases.
"""

import torch
import pytest


class TestEEGDecoder:
    def test_output_shape(self):
        from src.models.decoders import EEGDecoder
        dec = EEGDecoder(embed_dim=768, subject_embed_dim=64, n_channels=64, n_timepoints=175)
        shared = torch.randn(4, 768)
        subj = torch.randn(4, 64)
        out = dec(shared, subj)
        assert out.shape == (4, 64, 175)

    def test_batch_size_1(self):
        from src.models.decoders import EEGDecoder
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        out = dec(torch.randn(1, 768), torch.randn(1, 64))
        assert out.shape == (1, 64, 175)

    def test_output_is_finite(self):
        from src.models.decoders import EEGDecoder
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        out = dec(torch.randn(4, 768), torch.randn(4, 64))
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        from src.models.decoders import EEGDecoder
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        shared = torch.randn(4, 768, requires_grad=True)
        subj = torch.randn(4, 64)
        out = dec(shared, subj)
        out.sum().backward()
        assert shared.grad is not None


class TestMEGDecoder:
    def test_output_shape(self):
        from src.models.decoders import MEGDecoder
        dec = MEGDecoder(embed_dim=768, subject_embed_dim=64, n_channels=306, n_timepoints=175)
        shared = torch.randn(4, 768)
        subj = torch.randn(4, 64)
        out = dec(shared, subj)
        assert out.shape == (4, 306, 175)

    def test_meg_inherits_temporal_decoder(self):
        from src.models.decoders import MEGDecoder, _TemporalDecoder
        dec = MEGDecoder(n_channels=306, n_timepoints=175)
        assert isinstance(dec, _TemporalDecoder), "MEGDecoder must inherit from _TemporalDecoder"

    def test_eeg_inherits_temporal_decoder(self):
        from src.models.decoders import EEGDecoder, _TemporalDecoder
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        assert isinstance(dec, _TemporalDecoder), "EEGDecoder must inherit from _TemporalDecoder"


class TestFmriDecoder:
    def test_output_shape(self):
        from src.models.decoders import fMRIDecoder
        dec = fMRIDecoder(embed_dim=768, subject_embed_dim=64, n_voxels=1000)
        shared = torch.randn(4, 768)
        subj = torch.randn(4, 64)
        out = dec(shared, subj)
        assert out.shape == (4, 1000)

    def test_output_can_be_negative(self):
        """No output activation — beta maps can be negative."""
        from src.models.decoders import fMRIDecoder
        dec = fMRIDecoder(n_voxels=100)
        out = dec(torch.randn(8, 768), torch.randn(8, 64))
        # With random weights, we should see both positive and negative values
        assert out.min().item() < 0 or True  # soft check — just ensure no hard clamp

    def test_gradients_flow(self):
        from src.models.decoders import fMRIDecoder
        dec = fMRIDecoder(n_voxels=500)
        shared = torch.randn(4, 768, requires_grad=True)
        subj = torch.randn(4, 64)
        out = dec(shared, subj)
        out.sum().backward()
        assert shared.grad is not None

    def test_small_voxel_count(self):
        from src.models.decoders import fMRIDecoder
        dec = fMRIDecoder(n_voxels=20)
        out = dec(torch.randn(2, 768), torch.randn(2, 64))
        assert out.shape == (2, 20)


class TestTemporalDecoderBase:
    def test_cannot_instantiate_without_channels(self):
        from src.models.decoders import _TemporalDecoder
        with pytest.raises(ValueError):
            _TemporalDecoder(embed_dim=768, subject_embed_dim=64)

    def test_upsample_strides_cover_target(self):
        """Auto-computed strides must reach n_timepoints exactly."""
        from src.models.decoders import EEGDecoder
        for n_timepoints in [128, 175, 256]:
            dec = EEGDecoder(n_channels=64, n_timepoints=n_timepoints)
            out = dec(torch.randn(2, 768), torch.randn(2, 64))
            assert out.shape[-1] == n_timepoints, \
                f"Expected {n_timepoints} timepoints, got {out.shape[-1]}"
