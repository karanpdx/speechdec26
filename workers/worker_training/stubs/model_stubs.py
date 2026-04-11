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
        self._dummy = nn.Linear(1, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=(1, 2), keepdim=True)
        return self._dummy(pooled.view(x.shape[0], 1))


class MEGEncoder(nn.Module):
    """Stub: (B, n_channels, n_timepoints) → (B, embed_dim)"""
    def __init__(self, n_channels=306, n_timepoints=175, embed_dim=768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(1, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=(1, 2), keepdim=True)
        return self._dummy(pooled.view(x.shape[0], 1))


class fMRIEncoder(nn.Module):
    """Stub: (B, n_voxels) → (B, embed_dim)"""
    def __init__(self, n_voxels=1000, embed_dim=768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(1, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1, keepdim=True)
        return self._dummy(pooled)


class SharedEmbeddingProjector(nn.Module):
    """Stub: (B, 768) → (B, embed_dim)"""
    def __init__(self, bert_dim=768, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(bert_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self._dummy(x)


class SubjectEmbedding(nn.Module):
    """Stub: (B,) → (B, subject_embed_dim)"""
    def __init__(self, n_subjects=20, subject_embed_dim=64):
        super().__init__()
        self.subject_embed_dim = subject_embed_dim
        self.embedding = nn.Embedding(n_subjects, subject_embed_dim)

    def forward(self, subject_ids: Tensor) -> Tensor:
        return self.embedding(subject_ids)

    def get_mean_embedding(self) -> Tensor:
        return self.embedding.weight.mean(dim=0, keepdim=True)


class EEGDecoder(nn.Module):
    """Stub: (B, embed_dim), (B, subject_embed_dim) → (B, n_channels, n_timepoints)"""
    def __init__(self, embed_dim=768, subject_embed_dim=64, n_channels=64, n_timepoints=175):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.channel_proj = nn.Linear(embed_dim + subject_embed_dim, n_channels)
        self.time_basis = nn.Parameter(torch.randn(n_timepoints) * 0.02)

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        fused = torch.cat([shared_emb, subject_emb], dim=-1)
        channel_out = self.channel_proj(fused)
        return channel_out.unsqueeze(-1) * self.time_basis.view(1, 1, self.n_timepoints)


class MEGDecoder(nn.Module):
    """Stub: (B, embed_dim), (B, subject_embed_dim) → (B, n_channels, n_timepoints)"""
    def __init__(self, embed_dim=768, subject_embed_dim=64, n_channels=306, n_timepoints=175):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.channel_proj = nn.Linear(embed_dim + subject_embed_dim, n_channels)
        self.time_basis = nn.Parameter(torch.randn(n_timepoints) * 0.02)

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        fused = torch.cat([shared_emb, subject_emb], dim=-1)
        channel_out = self.channel_proj(fused)
        return channel_out.unsqueeze(-1) * self.time_basis.view(1, 1, self.n_timepoints)


class fMRIDecoder(nn.Module):
    """Stub: (B, embed_dim), (B, subject_embed_dim) → (B, n_voxels)"""
    def __init__(self, embed_dim=768, subject_embed_dim=64, n_voxels=1000):
        super().__init__()
        self.n_voxels = n_voxels
        self.proj = nn.Linear(embed_dim + subject_embed_dim, n_voxels)

    def forward(self, shared_emb: Tensor, subject_emb: Tensor) -> Tensor:
        fused = torch.cat([shared_emb, subject_emb], dim=-1)
        return self.proj(fused)
