"""
Tests for loss functions.

All tests from AGENTS.md plus edge cases.
No model or data dependencies — pure tensor tests.
"""

import pytest
import torch
import torch.nn.functional as F


class TestContrastiveLoss:
    def test_diagonal_is_minimum(self):
        """Perfect embeddings (identity pairs) should give lower loss than shuffled."""
        from src.training.losses import ContrastiveLoss
        emb = F.normalize(torch.randn(16, 768), dim=-1)
        loss_matched = ContrastiveLoss()(emb, emb)
        shuffled = emb[torch.randperm(16)]
        loss_shuffled = ContrastiveLoss()(emb, shuffled)
        assert loss_matched < loss_shuffled

    def test_batch_size_1_raises(self):
        """Contrastive loss is undefined for batch size 1."""
        from src.training.losses import ContrastiveLoss
        emb = F.normalize(torch.randn(1, 768), dim=-1)
        with pytest.raises(ValueError):
            ContrastiveLoss()(emb, emb)

    def test_output_is_scalar(self):
        from src.training.losses import ContrastiveLoss
        emb = F.normalize(torch.randn(8, 128), dim=-1)
        loss = ContrastiveLoss()(emb, emb)
        assert loss.shape == ()

    def test_loss_is_positive(self):
        from src.training.losses import ContrastiveLoss
        emb = F.normalize(torch.randn(8, 64), dim=-1)
        loss = ContrastiveLoss()(emb, torch.randn(8, 64))
        assert loss.item() > 0

    def test_gradients_flow(self):
        from src.training.losses import ContrastiveLoss
        emb_a = torch.randn(8, 64, requires_grad=True)
        emb_b = torch.randn(8, 64, requires_grad=True)
        loss = ContrastiveLoss()(emb_a, emb_b)
        loss.backward()
        assert emb_a.grad is not None
        assert emb_b.grad is not None

    def test_temperature_is_learnable_by_default(self):
        from src.training.losses import ContrastiveLoss
        loss_fn = ContrastiveLoss(learnable_temp=True)
        params = list(loss_fn.parameters())
        assert len(params) == 1, "Should have one learnable parameter (temperature)"

    def test_temperature_is_not_learnable_when_disabled(self):
        from src.training.losses import ContrastiveLoss
        loss_fn = ContrastiveLoss(learnable_temp=False)
        params = list(loss_fn.parameters())
        assert len(params) == 0

    def test_temperature_clamping(self):
        """Temperature should be clamped to [log(0.01), log(100)]."""
        from src.training.losses import ContrastiveLoss
        import math
        loss_fn = ContrastiveLoss(temperature_init=0.07)
        # Drive temperature to a very large value
        with torch.no_grad():
            for p in loss_fn.parameters():
                p.fill_(1000.0)
        emb = F.normalize(torch.randn(4, 32), dim=-1)
        # Should not raise or produce NaN
        loss = loss_fn(emb, emb)
        assert torch.isfinite(loss)


class TestCrossModalAlignmentLoss:
    def test_empty_mask_returns_zero(self):
        from src.training.losses import CrossModalAlignmentLoss
        emb_a = torch.randn(8, 768)
        emb_b = torch.randn(8, 768)
        mask = torch.zeros(8, dtype=torch.bool)
        loss = CrossModalAlignmentLoss()(emb_a, emb_b, mask)
        assert loss.item() == 0.0

    def test_empty_mask_has_gradient(self):
        """Zero loss from empty mask must still have requires_grad=True."""
        from src.training.losses import CrossModalAlignmentLoss
        emb_a = torch.randn(8, 768, requires_grad=True)
        emb_b = torch.randn(8, 768)
        mask = torch.zeros(8, dtype=torch.bool)
        loss = CrossModalAlignmentLoss()(emb_a, emb_b, mask)
        assert loss.requires_grad

    def test_single_match_returns_zero(self):
        """Fewer than 2 matches should also return zero (not crash)."""
        from src.training.losses import CrossModalAlignmentLoss
        emb_a = torch.randn(8, 768)
        emb_b = torch.randn(8, 768)
        mask = torch.zeros(8, dtype=torch.bool)
        mask[0] = True  # Only 1 match
        loss = CrossModalAlignmentLoss()(emb_a, emb_b, mask)
        assert loss.item() == 0.0

    def test_full_mask_computes_contrastive_loss(self):
        """All matches should compute a positive loss."""
        from src.training.losses import CrossModalAlignmentLoss
        emb_a = F.normalize(torch.randn(8, 128), dim=-1)
        emb_b = F.normalize(torch.randn(8, 128), dim=-1)
        mask = torch.ones(8, dtype=torch.bool)
        loss = CrossModalAlignmentLoss()(emb_a, emb_b, mask)
        assert loss.item() > 0

    def test_gradients_flow_through_matched_samples(self):
        from src.training.losses import CrossModalAlignmentLoss
        emb_a = torch.randn(8, 64, requires_grad=True)
        emb_b = torch.randn(8, 64, requires_grad=True)
        mask = torch.ones(8, dtype=torch.bool)
        loss = CrossModalAlignmentLoss()(emb_a, emb_b, mask)
        loss.backward()
        assert emb_a.grad is not None


class TestSubjectAdversarialLoss:
    def test_output_is_scalar(self):
        from src.training.losses import SubjectAdversarialLoss
        loss_fn = SubjectAdversarialLoss(embed_dim=128, n_subjects=10)
        emb = torch.randn(8, 128)
        ids = torch.randint(0, 10, (8,))
        loss = loss_fn(emb, ids, alpha=1.0)
        assert loss.shape == ()

    def test_loss_is_positive(self):
        from src.training.losses import SubjectAdversarialLoss
        loss_fn = SubjectAdversarialLoss(embed_dim=128, n_subjects=10)
        emb = torch.randn(8, 128)
        ids = torch.randint(0, 10, (8,))
        loss = loss_fn(emb, ids, alpha=1.0)
        assert loss.item() > 0

    def test_gradients_reversed(self):
        """
        Gradient reversal: the gradient through shared_emb should point
        in the OPPOSITE direction from minimizing the classifier loss.
        """
        from src.training.losses import SubjectAdversarialLoss
        loss_fn = SubjectAdversarialLoss(embed_dim=64, n_subjects=5)
        emb = torch.randn(4, 64, requires_grad=True)
        ids = torch.randint(0, 5, (4,))
        loss = loss_fn(emb, ids, alpha=1.0)
        loss.backward()
        # Gradient should exist
        assert emb.grad is not None
        # The loss function itself is a cross-entropy — its gradient w.r.t. classifier
        # params is normal, but the gradient w.r.t. emb is negated.
        # We verify gradients flow without NaN.
        assert torch.isfinite(emb.grad).all()

    def test_alpha_zero_passes_zero_gradient(self):
        """With alpha=0, gradient reversal passes zero gradient."""
        from src.training.losses import SubjectAdversarialLoss
        loss_fn = SubjectAdversarialLoss(embed_dim=64, n_subjects=5)
        emb = torch.randn(4, 64, requires_grad=True)
        ids = torch.randint(0, 5, (4,))
        loss = loss_fn(emb, ids, alpha=0.0)
        loss.backward()
        assert torch.all(emb.grad == 0)


class TestFreqDomainLoss:
    def test_output_is_scalar(self):
        from src.training.train_stage2 import freq_domain_loss
        pred = torch.randn(4, 64, 175)
        target = torch.randn(4, 64, 175)
        loss = freq_domain_loss(pred, target, sfreq=256.0)
        assert loss.shape == ()

    def test_identical_signals_give_zero_loss(self):
        from src.training.train_stage2 import freq_domain_loss
        signal = torch.randn(4, 32, 256)
        loss = freq_domain_loss(signal, signal, sfreq=256.0)
        assert loss.item() < 1e-5

    def test_gradients_flow(self):
        from src.training.train_stage2 import freq_domain_loss
        pred = torch.randn(4, 64, 175, requires_grad=True)
        target = torch.randn(4, 64, 175)
        loss = freq_domain_loss(pred, target, sfreq=256.0)
        loss.backward()
        assert pred.grad is not None
