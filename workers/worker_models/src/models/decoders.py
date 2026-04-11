"""
Decoder architectures for Stage 2 of the neural speech decoding pipeline.

Decoders reconstruct predicted neural signals from a shared embedding
conditioned on subject identity.

Classes:
    _TemporalDecoder  — Abstract base class for EEG/MEG decoders
    EEGDecoder        — Reconstructs EEG epochs from shared + subject embeddings
    MEGDecoder        — Reconstructs MEG epochs (inherits _TemporalDecoder)
    fMRIDecoder       — Reconstructs fMRI beta maps from shared + subject embeddings
"""

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class _TemporalDecoder(nn.Module):
    """
    Abstract base class for temporal signal decoders (EEG and MEG).

    Subclasses set n_channels and n_timepoints; this base class implements
    the shared upsampling architecture.

    Input:  shared_emb (batch, embed_dim), subject_emb (batch, subject_embed_dim)
    Output: (batch, n_channels, n_timepoints)

    Architecture:
        1. Concat → (batch, embed_dim + subject_embed_dim)
        2. Linear → 1024 + ReLU
        3. Linear → 2048 + ReLU
        4. Reshape → (batch, n_channels, 2048 // n_channels)
        5. ConvTranspose1d blocks (strides auto-computed from n_timepoints) + BatchNorm + ReLU
        6. Final Conv1d(n_channels, n_channels, 1)
        7. Output: (batch, n_channels, n_timepoints)

    Note:
        The number of transposed conv layers and their strides are computed
        automatically in __init__ based on n_timepoints. No hardcoding.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        subject_embed_dim: int = 64,
        n_channels: int = None,
        n_timepoints: int = None,
    ):
        """
        Args:
            embed_dim: Dimensionality of the shared embedding input.
            subject_embed_dim: Dimensionality of the subject embedding input.
            n_channels: Number of output channels.
            n_timepoints: Number of output time samples.

        Raises:
            ValueError: If n_channels or n_timepoints is None.
        """
        super().__init__()
        if n_channels is None or n_timepoints is None:
            raise ValueError("n_channels and n_timepoints must be specified.")
        raise NotImplementedError

    def _compute_upsample_strides(self, input_len: int, target_len: int) -> list[int]:
        """
        Compute ConvTranspose1d strides to upsample from input_len to target_len.

        Determines the number of layers and per-layer strides automatically.

        Args:
            input_len: Starting time dimension (2048 // n_channels).
            target_len: Target n_timepoints.

        Returns:
            List of integer strides, one per transposed conv layer.
        """
        raise NotImplementedError

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        """
        Args:
            shared_emb:  (batch, embed_dim)
            subject_emb: (batch, subject_embed_dim)

        Returns:
            (batch, n_channels, n_timepoints)
        """
        raise NotImplementedError


class EEGDecoder(_TemporalDecoder):
    """
    Reconstructs an EEG epoch from a shared embedding and subject embedding.

    Input:  shared_emb (batch, embed_dim), subject_emb (batch, subject_embed_dim)
    Output: (batch, n_channels, n_timepoints)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        subject_embed_dim: int = 64,
        n_channels: int = 64,
        n_timepoints: int = 175,
    ):
        super().__init__(
            embed_dim=embed_dim,
            subject_embed_dim=subject_embed_dim,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
        )


class MEGDecoder(_TemporalDecoder):
    """
    Reconstructs a MEG epoch from a shared embedding and subject embedding.

    Identical architecture to EEGDecoder. Uses _TemporalDecoder base —
    not a copy-paste of EEGDecoder.

    Input:  shared_emb (batch, embed_dim), subject_emb (batch, subject_embed_dim)
    Output: (batch, n_channels, n_timepoints)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        subject_embed_dim: int = 64,
        n_channels: int = 306,
        n_timepoints: int = 175,
    ):
        super().__init__(
            embed_dim=embed_dim,
            subject_embed_dim=subject_embed_dim,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
        )


class fMRIDecoder(nn.Module):
    """
    Reconstructs an fMRI beta map from a shared embedding and subject embedding.

    MLP-only decoder (no temporal structure to reconstruct).
    No output activation — beta maps can be positive or negative.

    Input:  shared_emb (batch, embed_dim), subject_emb (batch, subject_embed_dim)
    Output: (batch, n_voxels)

    Architecture:
        Concat → Linear(embed_dim + subject_embed_dim, 1024) + ReLU + Dropout(0.3)
               → Linear(1024, 2048) + ReLU + Dropout(0.3)
               → Linear(2048, n_voxels)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        subject_embed_dim: int = 64,
        n_voxels: int = 1000,
        dropout: float = 0.3,
    ):
        """
        Args:
            embed_dim: Dimensionality of shared embedding input.
            subject_embed_dim: Dimensionality of subject embedding input.
            n_voxels: Number of output voxels (or PCA components).
            dropout: Dropout probability.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        """
        Args:
            shared_emb:  (batch, embed_dim)
            subject_emb: (batch, subject_embed_dim)

        Returns:
            (batch, n_voxels)  — no output activation
        """
        raise NotImplementedError
