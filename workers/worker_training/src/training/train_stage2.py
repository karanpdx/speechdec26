"""
Stage 2 training loop — decoder training with frozen Stage 1 encoders.

Loads a Stage 1 checkpoint, freezes all encoder weights, then trains
modality-specific decoders to reconstruct neural signals from word embeddings.

CRITICAL: Encoder weights are frozen and verified before any gradient step.

Entry point: scripts/train_stage2.py
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def load_stage1_checkpoint(checkpoint_path: str, models: dict, device: str) -> dict:
    """
    Load Stage 1 checkpoint and freeze all encoder weights.

    Verifies that all encoder parameters have requires_grad=False after loading.

    Args:
        checkpoint_path: Path to Stage 1 best.pt checkpoint.
        models: Dict with encoder keys (output of build_stage1_models()).
        device: torch device string.

    Returns:
        Dict of loaded encoder modules with frozen weights.

    Raises:
        AssertionError: If any encoder parameter still has requires_grad=True.
        FileNotFoundError: If checkpoint_path does not exist.
    """
    raise NotImplementedError


def verify_encoders_frozen(encoders: dict) -> None:
    """
    Assert that all encoder parameters have requires_grad=False.

    Call this before every training step in Stage 2.

    Args:
        encoders: Dict mapping modality name to encoder nn.Module.

    Raises:
        AssertionError: If any parameter requires grad.
    """
    raise NotImplementedError


def freq_domain_loss(pred: torch.Tensor, target: torch.Tensor, sfreq: float) -> torch.Tensor:
    """
    Frequency-domain loss penalizing PSD differences across neural frequency bands.

    Bands: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz).
    Bands are weighted equally. Computed on the time dimension (dim=-1).

    Args:
        pred:   (batch, n_channels, n_timepoints) predicted signal
        target: (batch, n_channels, n_timepoints) real signal
        sfreq:  sampling frequency in Hz

    Returns:
        Scalar loss (mean PSD MSE across all bands and channels).
    """
    raise NotImplementedError


def spatial_smoothness_loss(
    pred: torch.Tensor,
    adjacency_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial smoothness loss for fMRI: penalize L2 norm of differences
    between adjacent voxels in predicted beta maps.

    Args:
        pred: (batch, n_voxels) predicted beta maps
        adjacency_indices: (n_edges, 2) long tensor of adjacent voxel index pairs,
                           precomputed from voxel_coords at startup.

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def build_voxel_adjacency(voxel_coords: torch.Tensor, max_dist_mm: float = 6.0) -> torch.Tensor:
    """
    Build a sparse adjacency tensor from MNI voxel coordinates.

    Two voxels are adjacent if their Euclidean distance in MNI space <= max_dist_mm.

    Computed once at startup and reused for all batches.

    Args:
        voxel_coords: int32 (n_voxels, 3) MNI coordinates.
        max_dist_mm: Maximum distance to consider voxels adjacent.

    Returns:
        (n_edges, 2) long tensor of adjacent voxel index pairs.
    """
    raise NotImplementedError


def build_decoders(config: dict) -> dict:
    """
    Instantiate Stage 2 decoders.

    Args:
        config: Parsed train_stage2.yaml.

    Returns:
        Dict: {'eeg_decoder', 'meg_decoder', 'fmri_decoder'}
    """
    raise NotImplementedError


def train_one_epoch(
    encoders: dict,
    decoders: dict,
    projector,
    subject_emb,
    dataloader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    adjacency_indices: torch.Tensor = None,
) -> dict:
    """
    Run one full Stage 2 training epoch.

    For each batch:
        1. Forward frozen encoder → shared embedding
        2. Forward decoder → predicted signal
        3. MSE loss between predicted and real signal
        4. Frequency-domain loss for EEG/MEG
        5. Spatial smoothness loss for fMRI
        6. Total = mse_weight * MSE + freq_weight * freq + smooth_weight * smooth
        7. Backward (decoder params only), optimizer step

    Verifies encoders_frozen before step 1.

    Args:
        encoders: Frozen encoder dict from load_stage1_checkpoint().
        decoders: Output of build_decoders().
        projector: Frozen SharedEmbeddingProjector.
        subject_emb: SubjectEmbedding (trainable for Stage 2).
        dataloader: Training DataLoader.
        optimizer: Optimizer over decoder parameters only.
        config: Parsed config dict.
        adjacency_indices: Precomputed voxel adjacency for spatial smoothness loss.

    Returns:
        Dict of mean epoch losses per modality.
    """
    raise NotImplementedError


def train(config_path: str) -> None:
    """
    Full Stage 2 training run.

    Loads Stage 1 checkpoint, freezes encoders, trains decoders.
    Does NOT update any encoder weights at any point.

    Args:
        config_path: Path to configs/train_stage2.yaml.
    """
    raise NotImplementedError
