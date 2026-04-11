"""
Training loss functions for the neural speech decoding pipeline.

Classes:
    ContrastiveLoss           — CLIP-style symmetric contrastive loss (Stage 1)
    CrossModalAlignmentLoss   — Cross-modal contrastive loss for paired modalities
    SubjectAdversarialLoss    — Gradient reversal loss for subject de-identification
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss between neural and text embeddings.

    Inputs are L2-normalized inside the loss — do not pre-normalize.
    Temperature is learnable and clamped to [log(0.01), log(100)].

    Input:
        neural_emb: (batch, embed_dim)
        text_emb:   (batch, embed_dim)

    Returns:
        Scalar loss.

    Raises:
        ValueError: If batch size is 1 (contrastive loss undefined).
    """

    def __init__(self, temperature_init: float = 0.07, learnable_temp: bool = True):
        """
        Args:
            temperature_init: Initial temperature value.
            learnable_temp: If True, temperature is an nn.Parameter.
        """
        super().__init__()
        import math
        log_temp = math.log(temperature_init)
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.tensor(log_temp))
        else:
            self.register_buffer("log_temperature", torch.tensor(log_temp))

    def forward(self, neural_emb: Tensor, text_emb: Tensor) -> Tensor:
        """
        Args:
            neural_emb: (batch, embed_dim) — will be L2-normalized internally
            text_emb:   (batch, embed_dim) — will be L2-normalized internally

        Returns:
            Scalar loss (mean of neural→text and text→neural cross-entropies).

        Raises:
            ValueError: If batch size == 1.
        """
        import math
        batch_size = neural_emb.shape[0]
        if batch_size == 1:
            raise ValueError("Contrastive loss is undefined for batch size 1.")

        # Clamp log_temperature to [log(0.01), log(100)]
        log_temp_clamped = self.log_temperature.clamp(math.log(0.01), math.log(100))
        temperature = log_temp_clamped.exp()

        # L2 normalize
        neural_emb = F.normalize(neural_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # Compute similarity matrix
        logits = (neural_emb @ text_emb.T) / temperature  # (batch, batch)

        labels = torch.arange(batch_size, device=neural_emb.device)
        loss_n2t = F.cross_entropy(logits, labels)
        loss_t2n = F.cross_entropy(logits.T, labels)
        return (loss_n2t + loss_t2n) / 2


class CrossModalAlignmentLoss(nn.Module):
    """
    Contrastive loss between two different modality embeddings that share stimulus labels.

    Only computes the loss over samples where shared_label_mask is True.
    Returns zero (with gradient) if fewer than 2 samples have matching labels.

    Inputs:
        emb_a:             (batch, embed_dim)
        emb_b:             (batch, embed_dim)
        shared_label_mask: (batch,) bool — True if sample has a match in both modalities

    Returns:
        Scalar loss.
    """

    def __init__(self, temperature_init: float = 0.07):
        super().__init__()
        import math
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        shared_label_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            emb_a: (batch, embed_dim)
            emb_b: (batch, embed_dim)
            shared_label_mask: (batch,) bool tensor

        Returns:
            Scalar loss, or torch.tensor(0.0, requires_grad=True) if < 2 matched samples.
        """
        import math
        n_matched = shared_label_mask.sum().item()
        if n_matched < 2:
            return torch.tensor(0.0, requires_grad=True, device=emb_a.device)

        a = emb_a[shared_label_mask]
        b = emb_b[shared_label_mask]

        log_temp_clamped = self.log_temperature.clamp(math.log(0.01), math.log(100))
        temperature = log_temp_clamped.exp()

        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        logits = (a @ b.T) / temperature
        labels = torch.arange(int(n_matched), device=emb_a.device)
        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.T, labels)
        return (loss_a2b + loss_b2a) / 2


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer.

    Forward: identity.
    Backward: negates and scales gradient by alpha.
    """

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.alpha * grad_output, None


class SubjectAdversarialLoss(nn.Module):
    """
    Gradient reversal loss that makes the shared embedding uninformative
    about subject identity.

    Contains a small subject classifier. The gradient reversal layer
    causes the encoder to be trained AGAINST this classifier — making
    the shared space subject-agnostic.

    Inputs:
        shared_emb:  (batch, embed_dim)
        subject_ids: (batch,) int — subject indices
        alpha:       float — gradient reversal scale (increase over training)

    Returns:
        Scalar cross-entropy loss of subject classifier (reversed in backward).
    """

    def __init__(self, embed_dim: int, n_subjects: int, hidden_dim: int = 128):
        """
        Args:
            embed_dim: Dimensionality of shared embedding input.
            n_subjects: Number of subjects (classifier output classes).
            hidden_dim: Hidden layer size of the subject classifier.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_subjects),
        )

    def forward(
        self,
        shared_emb: Tensor,
        subject_ids: Tensor,
        alpha: float = 1.0,
    ) -> Tensor:
        """
        Args:
            shared_emb:  (batch, embed_dim)
            subject_ids: (batch,) long tensor of subject indices
            alpha:       Gradient reversal scale (increase linearly over training,
                         typically 0 → 1 over all epochs)

        Returns:
            Scalar cross-entropy loss (reversed via GradientReversalFunction).
        """
        reversed_emb = GradientReversalFunction.apply(shared_emb, alpha)
        logits = self.classifier(reversed_emb)
        return F.cross_entropy(logits, subject_ids)
