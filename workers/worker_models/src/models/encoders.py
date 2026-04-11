"""
Encoder architectures for Stage 1 of the neural speech decoding pipeline.

All encoders project modality-specific neural signals into a shared embedding space
of dimensionality `embed_dim` (default 768, matching BERT).

Classes:
    EEGEncoder              — EEGNet-style encoder for EEG epochs
    MEGEncoder              — Identical to EEGEncoder, supports temporal weight sharing
    fMRIEncoder             — MLP encoder for fMRI beta maps
    SharedEmbeddingProjector — Projects BERT embeddings into the learned shared space
    SubjectEmbedding        — Per-subject embedding lookup table
"""

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class EEGEncoder(nn.Module):
    """
    EEGNet-style encoder for EEG epochs.

    Input:  (batch, n_channels, n_timepoints)
    Output: (batch, embed_dim)

    Architecture: depthwise spatial conv → depthwise temporal conv →
    avg pool → separable conv → avg pool → flatten → linear → LayerNorm.
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        embed_dim: int = 768,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
        share_temporal: bool = False,
    ):
        """
        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples per epoch.
            embed_dim: Output embedding dimensionality.
            F1: Number of temporal filters.
            D: Depth multiplier for depthwise spatial conv.
            F2: Number of pointwise filters in separable conv.
            dropout: Dropout probability.
            share_temporal: If True, temporal conv weights can be shared with MEGEncoder.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints)

        Returns:
            (batch, embed_dim)
        """
        raise NotImplementedError

    def share_temporal_weights(self, other: "MEGEncoder") -> None:
        """
        Share temporal conv weights with another encoder.

        After calling this, both encoders reference the same nn.Parameter
        object for temporal filter weights. Modifying one will affect the other.

        Args:
            other: MEGEncoder instance to share weights with.

        Raises:
            RuntimeError: If either encoder was not created with share_temporal=True.
        """
        raise NotImplementedError


class MEGEncoder(nn.Module):
    """
    EEGNet-style encoder for MEG epochs.

    Identical architecture to EEGEncoder but accepts different n_channels.
    Supports optional temporal weight sharing with EEGEncoder.

    Input:  (batch, n_channels, n_timepoints)
    Output: (batch, embed_dim)
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        embed_dim: int = 768,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
        share_temporal: bool = False,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints)

        Returns:
            (batch, embed_dim)
        """
        raise NotImplementedError


class fMRIEncoder(nn.Module):
    """
    MLP encoder for fMRI beta maps.

    Input:  (batch, n_voxels)
    Output: (batch, embed_dim)

    Architecture: Linear(n_voxels, 2048) + ReLU + Dropout
                → Linear(2048, 1024) + ReLU + Dropout
                → Linear(1024, embed_dim) + LayerNorm
    """

    def __init__(self, n_voxels: int, embed_dim: int = 768, dropout: float = 0.3):
        """
        Args:
            n_voxels: Number of input voxels (or PCA components).
            embed_dim: Output embedding dimensionality.
            dropout: Dropout probability.

        Note:
            Logs a warning if n_voxels > 10000 recommending PCA preprocessing.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_voxels)

        Returns:
            (batch, embed_dim)
        """
        raise NotImplementedError


class SharedEmbeddingProjector(nn.Module):
    """
    Projects BERT embeddings into the learned shared embedding space.

    Used as the target embedding source during contrastive training (Stage 1)
    and as the entry point for Stage 2 (word → shared space → decoder).

    BERT's space and the learned shared space are related but not identical.
    This projector bridges them and is trained jointly with the encoders.

    Input:  (batch, bert_dim=768)
    Output: (batch, embed_dim)

    Architecture: Linear(bert_dim, embed_dim) + ReLU + Linear(embed_dim, embed_dim) + LayerNorm
    """

    def __init__(self, bert_dim: int = 768, embed_dim: int = 768):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, bert_dim)

        Returns:
            (batch, embed_dim)
        """
        raise NotImplementedError


class SubjectEmbedding(nn.Module):
    """
    Per-subject embedding lookup table.

    Used to condition Stage 2 decoders on subject identity.

    Input:  subject_ids: (batch,) long tensor of subject indices
    Output: (batch, subject_embed_dim)
    """

    def __init__(self, n_subjects: int, subject_embed_dim: int = 64):
        """
        Args:
            n_subjects: Total number of subjects in the dataset.
            subject_embed_dim: Dimensionality of subject embeddings.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, subject_ids: Tensor) -> Tensor:
        """
        Args:
            subject_ids: (batch,) long tensor of subject indices in [0, n_subjects).

        Returns:
            (batch, subject_embed_dim)
        """
        raise NotImplementedError

    def get_mean_embedding(self) -> Tensor:
        """
        Returns the mean of all subject embeddings.

        Used as a generic prior for unseen subjects at inference time.

        Returns:
            (1, subject_embed_dim)
        """
        raise NotImplementedError
