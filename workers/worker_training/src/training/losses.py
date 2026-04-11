"""
Training loss functions for the neural speech decoding pipeline.

Classes:
    ContrastiveLoss           — CLIP-style symmetric contrastive loss (Stage 1)
    CrossModalAlignmentLoss   — Cross-modal contrastive loss for paired modalities
    SubjectAdversarialLoss    — Gradient reversal loss for subject de-identification
"""

import logging
import math

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
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature_init)))
        else:
            self.register_buffer('log_temp', torch.tensor(math.log(temperature_init)))

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
        if neural_emb.shape[0] == 1:
            raise ValueError("Contrastive loss is undefined for batch size 1")
        n = F.normalize(neural_emb, dim=-1)
        t = F.normalize(text_emb, dim=-1)
        temp = torch.clamp(self.log_temp, min=math.log(0.01), max=math.log(100)).exp()
        logits = temp * (n @ t.T)
        batch = neural_emb.shape[0]
        labels = torch.arange(batch, device=neural_emb.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss


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
        self.contrastive = ContrastiveLoss(temperature_init=temperature_init)

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
        n_matched = shared_label_mask.sum().item()
        if n_matched < 2:
            return torch.tensor(0.0, requires_grad=True)
        a = emb_a[shared_label_mask]
        b = emb_b[shared_label_mask]
        return self.contrastive(a, b)


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
        raise NotImplementedError

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
        raise NotImplementedError
