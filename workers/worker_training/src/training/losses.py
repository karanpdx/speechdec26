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
        log_temp = torch.tensor(temperature_init).log()
        if learnable_temp:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

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
        batch_size = neural_emb.shape[0]
        if batch_size == 1:
            raise ValueError("ContrastiveLoss requires batch_size > 1")

        # Clamp log temperature and get scale
        log_temp = self.log_temperature.clamp(
            torch.tensor(0.01).log().item(),
            torch.tensor(100.0).log().item(),
        )
        scale = log_temp.exp()

        # L2 normalize
        n = F.normalize(neural_emb, dim=-1)
        t = F.normalize(text_emb, dim=-1)

        # Similarity matrix (B, B)
        logits = (n @ t.T) / scale

        labels = torch.arange(batch_size, device=neural_emb.device)
        loss_n2t = F.cross_entropy(logits, labels)
        loss_t2n = F.cross_entropy(logits.T, labels)
        return (loss_n2t + loss_t2n) / 2.0


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
        log_temp = torch.tensor(temperature_init).log()
        self.log_temperature = nn.Parameter(log_temp)

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
        mask = shared_label_mask.bool()
        a = emb_a[mask]
        b = emb_b[mask]
        if a.shape[0] < 2:
            return torch.tensor(0.0, device=emb_a.device, requires_grad=True)

        scale = self.log_temperature.clamp(
            torch.tensor(0.01).log().item(),
            torch.tensor(100.0).log().item(),
        ).exp()

        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = (a @ b.T) / scale
        labels = torch.arange(a.shape[0], device=emb_a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2.0


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
        return F.cross_entropy(logits, subject_ids.long())
