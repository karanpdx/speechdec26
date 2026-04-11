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

    Subclasses (EEGDecoder, MEGDecoder) specify n_channels and n_timepoints;
    this base class implements the full shared upsampling architecture.

    Input:  shared_emb  (batch, embed_dim)
            subject_emb (batch, subject_embed_dim)
    Output: (batch, n_channels, n_timepoints)

    Architecture:
        1. Concat shared_emb + subject_emb → (B, embed_dim + subject_embed_dim)
        2. Linear(embed_dim + subject_embed_dim, 1024) + ReLU
        3. Linear(1024, 2048) + ReLU
        4. Reshape → (B, n_channels, 2048 // n_channels)
        5. ConvTranspose1d blocks with auto-computed strides + BatchNorm + ReLU
        6. Final Conv1d(n_channels, n_channels, kernel_size=1) channel mixer
        7. Crop/pad to exactly n_timepoints if needed

    Notes:
        - The number of transposed conv layers and their strides are computed
          in __init__ from n_timepoints. Nothing is hardcoded.
        - If 2048 is not evenly divisible by n_channels, the reshape pads with
          zeros and the final Conv1d absorbs any mismatch.
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
            embed_dim:          Dimensionality of the shared embedding input.
            subject_embed_dim:  Dimensionality of the subject embedding input.
            n_channels:         Number of output channels.
            n_timepoints:       Number of output time samples.

        Raises:
            ValueError: If n_channels or n_timepoints is None.
        """
        super().__init__()
        if n_channels is None or n_timepoints is None:
            raise ValueError(
                "n_channels and n_timepoints must both be specified. "
                f"Got n_channels={n_channels}, n_timepoints={n_timepoints}."
            )

        self.embed_dim = embed_dim
        self.subject_embed_dim = subject_embed_dim
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        in_dim = embed_dim + subject_embed_dim

        # MLP stem: compress to a spatial-temporal seed
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )

        # Determine starting time length after reshape.
        # We want shape (B, n_channels, t0) where n_channels * t0 = 2048.
        # If 2048 % n_channels != 0 we round up and store the true pad width.
        self._t0 = math.ceil(2048 / n_channels)
        self._mlp_out = n_channels * self._t0  # may be > 2048 if padding needed

        if self._mlp_out != 2048:
            # Replace the last Linear so its output matches the padded size.
            self.mlp[-2] = nn.Linear(1024, self._mlp_out)

        # Auto-compute ConvTranspose1d strides.
        strides = self._compute_upsample_strides(self._t0, n_timepoints)
        logger.info(
            "_TemporalDecoder: n_channels=%d, n_timepoints=%d, t0=%d, strides=%s",
            n_channels, n_timepoints, self._t0, strides,
        )

        # Build the transposed-conv stack.
        layers = []
        in_ch = n_channels
        for stride in strides:
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=in_ch,      # keep channel count stable through upsampling
                    kernel_size=stride * 2,  # kernel >= stride to avoid gaps
                    stride=stride,
                    padding=stride // 2,
                )
            )
            layers.append(nn.BatchNorm1d(in_ch))
            layers.append(nn.ReLU())

        self.upsample = nn.Sequential(*layers)

        # Channel mixer: 1×1 conv to allow cross-channel information flow.
        self.channel_mixer = nn.Conv1d(n_channels, n_channels, kernel_size=1)

    def _compute_upsample_strides(self, input_len: int, target_len: int) -> list:
        """
        Compute ConvTranspose1d strides to upsample from input_len to target_len.

        Strategy: repeatedly apply the largest integer stride ≤ 8 that makes
        progress, until the cumulative product ≥ target_len / input_len.
        Uses small integer strides (2–4) to keep the stack shallow and stable.

        Args:
            input_len:  Starting time dimension (t0 = ceil(2048 / n_channels)).
            target_len: Desired output n_timepoints.

        Returns:
            List of integer strides, one per ConvTranspose1d layer.
            The product of strides × input_len will be ≥ target_len;
            the forward pass crops to exactly target_len.
        """
        if target_len <= input_len:
            # No upsampling needed; return a stride-1 identity layer.
            return [1]

        ratio = target_len / input_len
        strides = []
        accumulated = 1.0
        while accumulated < ratio:
            # Prefer stride 4, fall back to 2 for fine-grained control.
            stride = 4 if accumulated * 4 <= ratio * 1.5 else 2
            strides.append(stride)
            accumulated *= stride
            if len(strides) > 10:
                # Safety valve: one final large stride to close any remaining gap.
                remaining = math.ceil(ratio / accumulated)
                if remaining > 1:
                    strides.append(remaining)
                break

        return strides

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        """
        Args:
            shared_emb:  (batch, embed_dim)
            subject_emb: (batch, subject_embed_dim)

        Returns:
            (batch, n_channels, n_timepoints)
        """
        if torch.is_grad_enabled():
            assert shared_emb.ndim == 2, (
                f"shared_emb must be 2D, got {shared_emb.ndim}D"
            )
            assert subject_emb.ndim == 2, (
                f"subject_emb must be 2D, got {subject_emb.ndim}D"
            )
            assert shared_emb.shape[1] == self.embed_dim, (
                f"Expected embed_dim={self.embed_dim}, got {shared_emb.shape[1]}"
            )
            assert subject_emb.shape[1] == self.subject_embed_dim, (
                f"Expected subject_embed_dim={self.subject_embed_dim}, "
                f"got {subject_emb.shape[1]}"
            )
            assert shared_emb.shape[0] == subject_emb.shape[0], (
                "Batch size mismatch between shared_emb and subject_emb: "
                f"{shared_emb.shape[0]} vs {subject_emb.shape[0]}"
            )

        # 1. Concat and project through MLP stem.
        x = torch.cat([shared_emb, subject_emb], dim=1)   # (B, in_dim)
        x = self.mlp(x)                                    # (B, mlp_out)

        # 2. Reshape into (B, n_channels, t0).
        B = x.shape[0]
        x = x.view(B, self.n_channels, self._t0)           # (B, C, t0)

        # 3. Upsample through ConvTranspose1d blocks.
        x = self.upsample(x)                               # (B, C, t_upsampled)

        # 4. Channel mixer.
        x = self.channel_mixer(x)                          # (B, C, t_upsampled)

        # 5. Crop or pad to exactly n_timepoints.
        t = x.shape[2]
        if t > self.n_timepoints:
            x = x[:, :, : self.n_timepoints]
        elif t < self.n_timepoints:
            pad = self.n_timepoints - t
            x = torch.nn.functional.pad(x, (0, pad))

        return x   # (B, n_channels, n_timepoints)


class EEGDecoder(_TemporalDecoder):
    """
    Reconstructs an EEG epoch from a shared embedding and subject embedding.

    Input:  shared_emb  (batch, embed_dim)
            subject_emb (batch, subject_embed_dim)
    Output: (batch, n_channels, n_timepoints)

    Delegates entirely to _TemporalDecoder — no additional logic.
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

    Identical architecture to EEGDecoder via the shared _TemporalDecoder base.
    Default n_channels=306 reflects typical Elekta/MEGIN sensor count.

    Input:  shared_emb  (batch, embed_dim)
            subject_emb (batch, subject_embed_dim)
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

    MLP-only decoder — no temporal structure to reconstruct.
    No output activation: beta maps can be positive or negative.

    Input:  shared_emb  (batch, embed_dim)
            subject_emb (batch, subject_embed_dim)
    Output: (batch, n_voxels)

    Architecture:
        Concat → (B, embed_dim + subject_embed_dim)
        Linear(embed_dim + subject_embed_dim, 1024) + ReLU + Dropout(0.3)
        Linear(1024, 2048)  + ReLU + Dropout(0.3)
        Linear(2048, n_voxels)   ← no activation

    Assumptions:
        - n_voxels may be the raw voxel count or PCA-reduced dimensionality.
        - Output is unbounded; use MSE loss during training.
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
            embed_dim:          Dimensionality of shared embedding input.
            subject_embed_dim:  Dimensionality of subject embedding input.
            n_voxels:           Number of output voxels (or PCA components).
            dropout:            Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.subject_embed_dim = subject_embed_dim
        self.n_voxels = n_voxels

        in_dim = embed_dim + subject_embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, n_voxels),
            # No output activation — beta maps can be positive or negative.
        )

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        """
        Args:
            shared_emb:  (batch, embed_dim)
            subject_emb: (batch, subject_embed_dim)

        Returns:
            (batch, n_voxels)  — no output activation
        """
        if torch.is_grad_enabled():
            assert shared_emb.ndim == 2, (
                f"shared_emb must be 2D, got {shared_emb.ndim}D"
            )
            assert subject_emb.ndim == 2, (
                f"subject_emb must be 2D, got {subject_emb.ndim}D"
            )
            assert shared_emb.shape[1] == self.embed_dim, (
                f"Expected embed_dim={self.embed_dim}, got {shared_emb.shape[1]}"
            )
            assert subject_emb.shape[1] == self.subject_embed_dim, (
                f"Expected subject_embed_dim={self.subject_embed_dim}, "
                f"got {subject_emb.shape[1]}"
            )
            assert shared_emb.shape[0] == subject_emb.shape[0], (
                "Batch size mismatch: "
                f"{shared_emb.shape[0]} vs {subject_emb.shape[0]}"
            )

        x = torch.cat([shared_emb, subject_emb], dim=1)
        return self.net(x)
