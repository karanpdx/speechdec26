"""
Tests for decoder architectures in src/models/decoders.py.

Covers:
    - Output shapes for EEGDecoder, MEGDecoder, fMRIDecoder
    - _TemporalDecoder inheritance (MEGDecoder must NOT be a copy-paste of EEGDecoder)
    - Auto-computed stride logic in _compute_upsample_strides
    - Edge cases: batch size 1, various n_channels / n_timepoints combos
    - Expected failures: missing n_channels / n_timepoints, shape mismatches
    - Gradient flow
"""

import pytest
import torch

from src.models.decoders import (
    EEGDecoder,
    MEGDecoder,
    _TemporalDecoder,
    fMRIDecoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 768
SUBJ_DIM = 64


def make_inputs(batch: int = 4, embed_dim: int = EMBED_DIM, subj_dim: int = SUBJ_DIM):
    shared = torch.randn(batch, embed_dim)
    subj = torch.randn(batch, subj_dim)
    return shared, subj


# ---------------------------------------------------------------------------
# EEGDecoder
# ---------------------------------------------------------------------------


class TestEEGDecoder:
    def test_output_shape_standard(self):
        """Happy path from AGENTS.md spec."""
        dec = EEGDecoder(embed_dim=768, subject_embed_dim=64, n_channels=64, n_timepoints=175)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.shape == (4, 64, 175), f"Expected (4, 64, 175), got {out.shape}"

    def test_output_shape_small_batch(self):
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        shared, subj = make_inputs(batch=1)
        out = dec(shared, subj)
        assert out.shape == (1, 64, 175)

    def test_output_shape_custom_channels(self):
        dec = EEGDecoder(n_channels=32, n_timepoints=256)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.shape == (4, 32, 256)

    def test_output_is_float32(self):
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.dtype == torch.float32

    def test_output_no_nan(self):
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_gradients_flow(self):
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        shared = torch.randn(4, 768, requires_grad=True)
        subj = torch.randn(4, 64, requires_grad=True)
        dec(shared, subj).sum().backward()
        assert shared.grad is not None
        assert subj.grad is not None
        assert not torch.isnan(shared.grad).any()

    def test_wrong_embed_dim_raises(self):
        dec = EEGDecoder(embed_dim=768, n_channels=64, n_timepoints=175)
        shared = torch.randn(4, 512)   # wrong dim
        subj = torch.randn(4, 64)
        with pytest.raises((AssertionError, RuntimeError)):
            dec(shared, subj)

    def test_wrong_subject_dim_raises(self):
        dec = EEGDecoder(subject_embed_dim=64, n_channels=64, n_timepoints=175)
        shared = torch.randn(4, 768)
        subj = torch.randn(4, 32)   # wrong dim
        with pytest.raises((AssertionError, RuntimeError)):
            dec(shared, subj)

    def test_batch_mismatch_raises(self):
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        shared = torch.randn(4, 768)
        subj = torch.randn(8, 64)   # different batch
        with pytest.raises((AssertionError, RuntimeError)):
            dec(shared, subj)

    def test_parameters_exist(self):
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        assert len(list(dec.parameters())) > 0


# ---------------------------------------------------------------------------
# MEGDecoder
# ---------------------------------------------------------------------------


class TestMEGDecoder:
    def test_output_shape_standard(self):
        """Happy path from AGENTS.md spec."""
        dec = MEGDecoder(embed_dim=768, subject_embed_dim=64, n_channels=306, n_timepoints=175)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.shape == (4, 306, 175), f"Expected (4, 306, 175), got {out.shape}"

    def test_output_shape_custom(self):
        dec = MEGDecoder(n_channels=204, n_timepoints=300)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.shape == (4, 204, 300)

    def test_output_no_nan(self):
        dec = MEGDecoder(n_channels=306, n_timepoints=175)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_inherits_temporal_decoder(self):
        """MEGDecoder must be a subclass of _TemporalDecoder (not a copy-paste)."""
        dec = MEGDecoder(n_channels=306, n_timepoints=175)
        assert isinstance(dec, _TemporalDecoder), (
            "MEGDecoder must inherit from _TemporalDecoder"
        )

    def test_eeg_decoder_also_inherits_temporal_decoder(self):
        """EEGDecoder must also be a subclass of _TemporalDecoder."""
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        assert isinstance(dec, _TemporalDecoder)

    def test_gradients_flow(self):
        dec = MEGDecoder(n_channels=306, n_timepoints=175)
        shared = torch.randn(4, 768, requires_grad=True)
        subj = torch.randn(4, 64, requires_grad=True)
        dec(shared, subj).sum().backward()
        assert shared.grad is not None


# ---------------------------------------------------------------------------
# _TemporalDecoder base: missing arguments
# ---------------------------------------------------------------------------


class TestTemporalDecoderBase:
    def test_missing_n_channels_raises(self):
        with pytest.raises(ValueError):
            _TemporalDecoder(n_channels=None, n_timepoints=175)

    def test_missing_n_timepoints_raises(self):
        with pytest.raises(ValueError):
            _TemporalDecoder(n_channels=64, n_timepoints=None)

    def test_both_missing_raises(self):
        with pytest.raises(ValueError):
            _TemporalDecoder()

    def test_compute_upsample_strides_product_reaches_target(self):
        """Product of strides must bring input_len to at least target_len."""
        dec = EEGDecoder(n_channels=64, n_timepoints=175)
        for t0, target in [(5, 175), (8, 256), (3, 100), (10, 10), (20, 20)]:
            strides = dec._compute_upsample_strides(t0, target)
            product = 1
            for s in strides:
                product *= s
            assert t0 * product >= target, (
                f"t0={t0}, target={target}: strides={strides}, "
                f"t0*product={t0*product} < target"
            )

    def test_forward_crop_to_exact_timepoints(self):
        """Output must have exactly n_timepoints regardless of upsampling overshoot."""
        for n_tp in [50, 100, 175, 200, 256]:
            dec = EEGDecoder(n_channels=16, n_timepoints=n_tp)
            shared, subj = make_inputs()
            out = dec(shared, subj)
            assert out.shape[2] == n_tp, (
                f"n_timepoints={n_tp}: got {out.shape[2]}"
            )


# ---------------------------------------------------------------------------
# fMRIDecoder
# ---------------------------------------------------------------------------


class TestFMRIDecoder:
    def test_output_shape_standard(self):
        """Happy path from AGENTS.md spec."""
        dec = fMRIDecoder(embed_dim=768, subject_embed_dim=64, n_voxels=1000)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.shape == (4, 1000), f"Expected (4, 1000), got {out.shape}"

    def test_output_shape_small_voxels(self):
        dec = fMRIDecoder(n_voxels=128)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.shape == (4, 128)

    def test_output_shape_large_voxels(self):
        dec = fMRIDecoder(n_voxels=5000)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.shape == (4, 5000)

    def test_output_is_float32(self):
        dec = fMRIDecoder(n_voxels=1000)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert out.dtype == torch.float32

    def test_output_no_nan(self):
        dec = fMRIDecoder(n_voxels=1000)
        shared, subj = make_inputs()
        out = dec(shared, subj)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_can_be_negative(self):
        """Beta maps are unbounded — output must not be clipped to positive."""
        torch.manual_seed(0)
        dec = fMRIDecoder(n_voxels=1000)
        # Run many batches; with no output activation, some values will be negative.
        any_negative = False
        for _ in range(10):
            shared, subj = make_inputs(batch=16)
            out = dec(shared, subj)
            if (out < 0).any():
                any_negative = True
                break
        assert any_negative, (
            "fMRIDecoder output was never negative — "
            "there may be an unwanted output activation."
        )

    def test_no_output_activation(self):
        """The last layer of fMRIDecoder must not be an activation module."""
        dec = fMRIDecoder(n_voxels=1000)
        last_layer = list(dec.net.children())[-1]
        activation_types = (
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
            torch.nn.ELU,
            torch.nn.GELU,
            torch.nn.Softmax,
        )
        assert not isinstance(last_layer, activation_types), (
            f"Last layer is {type(last_layer).__name__}; expected Linear (no activation)."
        )

    def test_gradients_flow(self):
        dec = fMRIDecoder(n_voxels=1000)
        shared = torch.randn(4, 768, requires_grad=True)
        subj = torch.randn(4, 64, requires_grad=True)
        dec(shared, subj).sum().backward()
        assert shared.grad is not None
        assert subj.grad is not None

    def test_wrong_embed_dim_raises(self):
        dec = fMRIDecoder(embed_dim=768, n_voxels=1000)
        shared = torch.randn(4, 512)   # wrong
        subj = torch.randn(4, 64)
        with pytest.raises((AssertionError, RuntimeError)):
            dec(shared, subj)

    def test_batch_mismatch_raises(self):
        dec = fMRIDecoder(n_voxels=1000)
        shared = torch.randn(4, 768)
        subj = torch.randn(8, 64)   # different batch
        with pytest.raises((AssertionError, RuntimeError)):
            dec(shared, subj)

    def test_single_batch_works(self):
        dec = fMRIDecoder(n_voxels=1000)
        shared, subj = make_inputs(batch=1)
        out = dec(shared, subj)
        assert out.shape == (1, 1000)

    def test_parameters_exist(self):
        dec = fMRIDecoder(n_voxels=1000)
        assert len(list(dec.parameters())) > 0
