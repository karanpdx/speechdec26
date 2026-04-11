"""
Encoder architectures for Stage 1 of the neural speech decoding pipeline.

All encoders project modality-specific neural signals into a shared embedding space
of dimensionality `embed_dim` (default 768, matching BERT).

Classes:
    EEGEncoder               — EEGNet-style encoder for EEG epochs
    MEGEncoder               — Identical to EEGEncoder, supports temporal weight sharing
    fMRIEncoder              — MLP encoder for fMRI beta maps
    SharedEmbeddingProjector — Projects BERT embeddings into the learned shared space
    SubjectEmbedding         — Per-subject embedding lookup table
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

    Architecture:
        1. Depthwise Conv2d spatial filter: (1, n_channels) kernel across channels
        2. Depthwise Conv1d temporal filter with depth multiplier D
        3. Average pooling (temporal)
        4. Separable Conv1d: F2 pointwise filters
        5. Average pooling (temporal)
        6. Flatten
        7. Linear → embed_dim
        8. LayerNorm

    Assumptions:
        - Input must have shape (batch, n_channels, n_timepoints).
        - n_timepoints should be large enough to survive two avg-pool operations.
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
            n_channels:     Number of EEG channels.
            n_timepoints:   Number of time samples per epoch.
            embed_dim:      Output embedding dimensionality.
            F1:             Number of temporal filters.
            D:              Depth multiplier for depthwise spatial conv.
            F2:             Number of pointwise filters in separable conv.
            dropout:        Dropout probability.
            share_temporal: If True, temporal conv weights can be shared with MEGEncoder
                            via share_temporal_weights().
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.embed_dim = embed_dim
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.share_temporal = share_temporal

        # --- Block 1: Temporal filter (Conv2d treats input as (B, 1, C, T)) ---
        # Learns F1 temporal filters across time; kernel spans full channel dim.
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, 25),   # 1 across channels, 25 across time (EEGNet default)
            padding=(0, 12),       # 'same' padding along time
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(F1)

        # --- Block 2: Spatial (depthwise) filter ---
        # Groups=F1 so each temporal filter gets its own spatial filter per channel.
        self.spatial_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),  # collapses the channel dimension
            groups=F1,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(F1 * D)
        self.spatial_act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))   # downsample time ×4
        self.drop1 = nn.Dropout(dropout)

        # --- Block 3: Separable Conv (depthwise + pointwise) ---
        F_in = F1 * D
        self.sep_depthwise = nn.Conv2d(
            in_channels=F_in,
            out_channels=F_in,
            kernel_size=(1, 15),   # temporal kernel
            padding=(0, 7),
            groups=F_in,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=F_in,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.sep_bn = nn.BatchNorm2d(F2)
        self.sep_act = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))   # downsample time ×8
        self.drop2 = nn.Dropout(dropout)

        # --- Compute flattened size after pooling ---
        # After pool1: T // 4, after pool2: T // 32. Spatial dim collapses to 1.
        t_after_pools = n_timepoints // 4 // 8
        flat_dim = F2 * 1 * max(t_after_pools, 1)

        # --- Projection head ---
        self.fc = nn.Linear(flat_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints)

        Returns:
            (batch, embed_dim)
        """
        if torch.is_grad_enabled():
            assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
            assert x.shape[1] == self.n_channels, (
                f"Expected {self.n_channels} channels, got {x.shape[1]}"
            )
            assert x.shape[2] == self.n_timepoints, (
                f"Expected {self.n_timepoints} timepoints, got {x.shape[2]}"
            )

        # Add channel dim: (B, 1, C, T)
        x = x.unsqueeze(1)

        # Block 1: temporal
        x = self.temporal_conv(x)        # (B, F1, C, T)
        x = self.temporal_bn(x)

        # Block 2: spatial (depthwise)
        x = self.spatial_conv(x)         # (B, F1*D, 1, T)
        x = self.spatial_bn(x)
        x = self.spatial_act(x)
        x = self.pool1(x)                # (B, F1*D, 1, T//4)
        x = self.drop1(x)

        # Block 3: separable
        x = self.sep_depthwise(x)        # (B, F1*D, 1, T//4)
        x = self.sep_pointwise(x)        # (B, F2,   1, T//4)
        x = self.sep_bn(x)
        x = self.sep_act(x)
        x = self.pool2(x)                # (B, F2,   1, T//32)
        x = self.drop2(x)

        # Flatten and project
        x = x.flatten(1)                 # (B, F2 * T//32)
        x = self.fc(x)                   # (B, embed_dim)
        x = self.norm(x)
        return x

    def share_temporal_weights(self, other: "MEGEncoder") -> None:
        """
        Share temporal conv weights with a MEGEncoder instance.

        After calling this, both encoders reference the same nn.Parameter
        object for temporal filter weights. Modifying one will affect the other.

        Args:
            other: MEGEncoder instance to share weights with.

        Raises:
            RuntimeError: If either encoder was not created with share_temporal=True.
        """
        if not self.share_temporal:
            raise RuntimeError(
                "This EEGEncoder was not created with share_temporal=True."
            )
        if not other.share_temporal:
            raise RuntimeError(
                "The target MEGEncoder was not created with share_temporal=True."
            )
        # Point other's temporal conv weight to this one's parameter
        other.temporal_conv.weight = self.temporal_conv.weight
        logger.info(
            "Temporal conv weights shared between EEGEncoder and MEGEncoder."
        )


class MEGEncoder(nn.Module):
    """
    EEGNet-style encoder for MEG epochs.

    Identical architecture to EEGEncoder but accepts a different n_channels.
    Supports optional temporal weight sharing with EEGEncoder via
    EEGEncoder.share_temporal_weights(meg_encoder).

    Input:  (batch, n_channels, n_timepoints)
    Output: (batch, embed_dim)

    Assumptions:
        - Input must have shape (batch, n_channels, n_timepoints).
        - share_temporal=True required if temporal weight sharing is desired.
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
            n_channels:     Number of MEG channels.
            n_timepoints:   Number of time samples per epoch.
            embed_dim:      Output embedding dimensionality.
            F1:             Number of temporal filters.
            D:              Depth multiplier for depthwise spatial conv.
            F2:             Number of pointwise filters in separable conv.
            dropout:        Dropout probability.
            share_temporal: If True, temporal conv weights are shareable with EEGEncoder.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.embed_dim = embed_dim
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.share_temporal = share_temporal

        # Identical architecture to EEGEncoder — only n_channels differs.
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, 25),
            padding=(0, 12),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(F1)

        self.spatial_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(F1 * D)
        self.spatial_act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        F_in = F1 * D
        self.sep_depthwise = nn.Conv2d(
            in_channels=F_in,
            out_channels=F_in,
            kernel_size=(1, 15),
            padding=(0, 7),
            groups=F_in,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=F_in,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.sep_bn = nn.BatchNorm2d(F2)
        self.sep_act = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        t_after_pools = n_timepoints // 4 // 8
        flat_dim = F2 * 1 * max(t_after_pools, 1)

        self.fc = nn.Linear(flat_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints)

        Returns:
            (batch, embed_dim)
        """
        if torch.is_grad_enabled():
            assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
            assert x.shape[1] == self.n_channels, (
                f"Expected {self.n_channels} channels, got {x.shape[1]}"
            )
            assert x.shape[2] == self.n_timepoints, (
                f"Expected {self.n_timepoints} timepoints, got {x.shape[2]}"
            )

        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.temporal_bn(x)

        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.spatial_act(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.sep_bn(x)
        x = self.sep_act(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(1)
        x = self.fc(x)
        x = self.norm(x)
        return x


class fMRIEncoder(nn.Module):
    """
    MLP encoder for fMRI beta maps.

    Input:  (batch, n_voxels)
    Output: (batch, embed_dim)

    Architecture:
        Linear(n_voxels, 2048) + ReLU + Dropout
        Linear(2048, 1024)     + ReLU + Dropout
        Linear(1024, embed_dim)
        LayerNorm(embed_dim)

    Assumptions:
        - Input is a flat vector of voxel activations (or PCA components).
        - Logs a warning if n_voxels > 10000, recommending PCA preprocessing.
    """

    def __init__(self, n_voxels: int, embed_dim: int = 768, dropout: float = 0.3):
        """
        Args:
            n_voxels:   Number of input voxels (or PCA components).
            embed_dim:  Output embedding dimensionality.
            dropout:    Dropout probability.
        """
        super().__init__()
        self.n_voxels = n_voxels
        self.embed_dim = embed_dim

        if n_voxels > 10_000:
            logger.warning(
                "fMRIEncoder received n_voxels=%d (> 10,000). "
                "Consider applying PCA preprocessing to reduce dimensionality "
                "and improve training stability.",
                n_voxels,
            )

        self.net = nn.Sequential(
            nn.Linear(n_voxels, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_voxels)

        Returns:
            (batch, embed_dim)
        """
        if torch.is_grad_enabled():
            assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
            assert x.shape[1] == self.n_voxels, (
                f"Expected {self.n_voxels} voxels, got {x.shape[1]}"
            )

        return self.net(x)


class SharedEmbeddingProjector(nn.Module):
    """
    Projects BERT embeddings into the learned shared embedding space.

    Used as the target embedding source during contrastive training (Stage 1)
    and as the entry point for Stage 2 (word → shared space → decoder).

    BERT's space and the learned shared space are related but not identical.
    This projector bridges them and is trained jointly with the encoders.

    Input:  (batch, bert_dim)   — typically bert_dim=768
    Output: (batch, embed_dim)

    Architecture:
        Linear(bert_dim, embed_dim) + ReLU
        Linear(embed_dim, embed_dim)
        LayerNorm(embed_dim)

    Assumptions:
        - Input is a BERT [CLS] token embedding (or mean-pooled BERT output).
        - BERT weights are frozen; only this projector is trained.
    """

    def __init__(self, bert_dim: int = 768, embed_dim: int = 768):
        """
        Args:
            bert_dim:   Dimensionality of input BERT embeddings.
            embed_dim:  Dimensionality of the output shared embedding space.
        """
        super().__init__()
        self.bert_dim = bert_dim
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, bert_dim)

        Returns:
            (batch, embed_dim)
        """
        if torch.is_grad_enabled():
            assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
            assert x.shape[1] == self.bert_dim, (
                f"Expected bert_dim={self.bert_dim}, got {x.shape[1]}"
            )

        return self.net(x)


class SubjectEmbedding(nn.Module):
    """
    Per-subject embedding lookup table.

    Used to condition Stage 2 decoders on subject identity.

    Input:  subject_ids: (batch,) — long tensor of subject indices in [0, n_subjects)
    Output: (batch, subject_embed_dim)

    Assumptions:
        - Subject indices must be in [0, n_subjects).
        - get_mean_embedding() can be used for unseen subjects at inference time.
    """

    def __init__(self, n_subjects: int, subject_embed_dim: int = 64):
        """
        Args:
            n_subjects:        Total number of subjects.
            subject_embed_dim: Dimensionality of each subject embedding.
        """
        super().__init__()
        self.n_subjects = n_subjects
        self.subject_embed_dim = subject_embed_dim
        self.embedding = nn.Embedding(n_subjects, subject_embed_dim)

    def forward(self, subject_ids: Tensor) -> Tensor:
        """
        Args:
            subject_ids: (batch,) long tensor of subject indices in [0, n_subjects).

        Returns:
            (batch, subject_embed_dim)
        """
        if torch.is_grad_enabled():
            assert subject_ids.ndim == 1, (
                f"Expected 1D subject_ids, got {subject_ids.ndim}D"
            )
            assert subject_ids.dtype in (torch.long, torch.int64), (
                f"subject_ids must be long tensor, got {subject_ids.dtype}"
            )

        return self.embedding(subject_ids)

    def get_mean_embedding(self) -> Tensor:
        """
        Returns the mean of all subject embeddings.

        Used as a generic prior for unseen subjects at inference time.

        Returns:
            (1, subject_embed_dim)
        """
        return self.embedding.weight.mean(dim=0, keepdim=True)
