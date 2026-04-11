"""
Model stubs for worker_training development.

These are drop-in replacements for the real models from worker_models.
They return correctly-shaped tensors with no actual computation.

Use these while Person 2 builds the real models. At integration,
replace all imports of stubs with imports from src.models.

Usage:
    from stubs.model_stubs import (
        EEGEncoder, MEGEncoder, fMRIEncoder,
        SharedEmbeddingProjector, SubjectEmbedding,
        EEGDecoder, MEGDecoder, fMRIDecoder,
    )
"""

import torch
import torch.nn as nn
from torch import Tensor


class EEGEncoder(nn.Module):
    """Stub: (B, n_channels, n_timepoints) → (B, embed_dim)"""
    def __init__(self, n_channels=64, n_timepoints=175, embed_dim=768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(1, 1)  # so optimizer has params

    def forward(self, x: Tensor) -> Tensor:
        # Return small random noise instead of zeros — zero vectors cause
        # degenerate (but not NaN) ContrastiveLoss outputs due to F.normalize(0).
        return torch.randn(x.shape[0], self.embed_dim, device=x.device) * 0.1


class MEGEncoder(nn.Module):
    """Stub: (B, n_channels, n_timepoints) → (B, embed_dim)"""
    def __init__(self, n_channels=306, n_timepoints=175, embed_dim=768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.randn(x.shape[0], self.embed_dim, device=x.device) * 0.1


class fMRIEncoder(nn.Module):
    """Stub: (B, n_voxels) → (B, embed_dim)"""
    def __init__(self, n_voxels=1000, embed_dim=768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.randn(x.shape[0], self.embed_dim, device=x.device) * 0.1


class SharedEmbeddingProjector(nn.Module):
    """Stub: (B, 768) → (B, embed_dim)"""
    def __init__(self, bert_dim=768, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.randn(x.shape[0], self.embed_dim, device=x.device) * 0.1


class SubjectEmbedding(nn.Module):
    """Stub: (B,) → (B, subject_embed_dim)"""
    def __init__(self, n_subjects=20, subject_embed_dim=64):
        super().__init__()
        self.subject_embed_dim = subject_embed_dim
        self._dummy = nn.Linear(1, 1)

    def forward(self, subject_ids: Tensor) -> Tensor:
        return torch.zeros(subject_ids.shape[0], self.subject_embed_dim, device=subject_ids.device)

    def get_mean_embedding(self) -> Tensor:
        return torch.zeros(1, self.subject_embed_dim)


class EEGDecoder(nn.Module):
    """Stub: (B, embed_dim), (B, subject_embed_dim) → (B, n_channels, n_timepoints)"""
    def __init__(self, embed_dim=768, subject_embed_dim=64, n_channels=64, n_timepoints=175):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self._dummy = nn.Linear(1, 1)

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        return torch.zeros(shared_emb.shape[0], self.n_channels, self.n_timepoints,
                           device=shared_emb.device)


class MEGDecoder(nn.Module):
    """Stub: (B, embed_dim), (B, subject_embed_dim) → (B, n_channels, n_timepoints)"""
    def __init__(self, embed_dim=768, subject_embed_dim=64, n_channels=306, n_timepoints=175):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self._dummy = nn.Linear(1, 1)

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        return torch.zeros(shared_emb.shape[0], self.n_channels, self.n_timepoints,
                           device=shared_emb.device)


class fMRIDecoder(nn.Module):
    """Stub: (B, embed_dim), (B, subject_embed_dim) → (B, n_voxels)"""
    def __init__(self, embed_dim=768, subject_embed_dim=64, n_voxels=1000):
        super().__init__()
        self.n_voxels = n_voxels
        self._dummy = nn.Linear(1, 1)

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        return torch.zeros(shared_emb.shape[0], self.n_voxels, device=shared_emb.device)
