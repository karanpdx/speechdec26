# Worker Models — Person 2

## Your Job

Implement all PyTorch model architectures: the three modality encoders, the shared embedding projector, the subject embedding lookup, and the three modality decoders. No training logic here — pure `nn.Module` definitions with tests.

**You have zero dependencies on any other worker.** You only need PyTorch. Test everything with `torch.randn(...)` inputs.

---

## What You Build

```
worker_models/
├── src/models/
│   ├── encoders.py   # EEGEncoder, MEGEncoder, fMRIEncoder,
│   │                 # SharedEmbeddingProjector, SubjectEmbedding
│   └── decoders.py   # _TemporalDecoder (base), EEGDecoder, MEGDecoder, fMRIDecoder
└── tests/
    ├── test_encoders.py
    └── test_decoders.py
```

---

## Interface Contract (what Person 3 imports from you)

Person 3 (training) will do:

```python
from src.models.encoders import (
    EEGEncoder, MEGEncoder, fMRIEncoder,
    SharedEmbeddingProjector, SubjectEmbedding,
)
from src.models.decoders import EEGDecoder, MEGDecoder, fMRIDecoder
```

**Every class must match these exact signatures:**

### Encoders

```python
EEGEncoder(n_channels: int, n_timepoints: int, embed_dim: int = 768,
           F1: int = 8, D: int = 2, F2: int = 16, dropout: float = 0.25,
           share_temporal: bool = False)
# forward(x: Tensor(B, n_channels, n_timepoints)) → Tensor(B, embed_dim)
# method: share_temporal_weights(other: MEGEncoder) → None

MEGEncoder(n_channels: int, n_timepoints: int, embed_dim: int = 768,
           F1: int = 8, D: int = 2, F2: int = 16, dropout: float = 0.25,
           share_temporal: bool = False)
# forward(x: Tensor(B, n_channels, n_timepoints)) → Tensor(B, embed_dim)

fMRIEncoder(n_voxels: int, embed_dim: int = 768, dropout: float = 0.3)
# forward(x: Tensor(B, n_voxels)) → Tensor(B, embed_dim)

SharedEmbeddingProjector(bert_dim: int = 768, embed_dim: int = 768)
# forward(x: Tensor(B, 768)) → Tensor(B, embed_dim)

SubjectEmbedding(n_subjects: int, subject_embed_dim: int = 64)
# forward(subject_ids: Tensor(B,) long) → Tensor(B, subject_embed_dim)
# method: get_mean_embedding() → Tensor(1, subject_embed_dim)
```

### Decoders

```python
EEGDecoder(embed_dim: int = 768, subject_embed_dim: int = 64,
           n_channels: int, n_timepoints: int)
# forward(shared_emb: Tensor(B, embed_dim),
#         subject_emb: Tensor(B, subject_embed_dim)) → Tensor(B, n_channels, n_timepoints)

MEGDecoder(embed_dim: int = 768, subject_embed_dim: int = 64,
           n_channels: int, n_timepoints: int)
# forward(shared_emb, subject_emb) → Tensor(B, n_channels, n_timepoints)
# NOTE: MEGDecoder must inherit from _TemporalDecoder — do not copy-paste EEGDecoder

fMRIDecoder(embed_dim: int = 768, subject_embed_dim: int = 64, n_voxels: int)
# forward(shared_emb: Tensor(B, embed_dim),
#         subject_emb: Tensor(B, subject_embed_dim)) → Tensor(B, n_voxels)
```

---

## Architecture Specs

### EEGEncoder / MEGEncoder (EEGNet-style)
1. Depthwise Conv2d across channels: `(1, n_channels) → (F1, n_channels)` spatial filter
2. Depthwise Conv1d across time: `F1 * D` temporal filters
3. Average pooling
4. Separable Conv1d: `F2` filters
5. Average pooling
6. Flatten
7. Linear → `embed_dim`
8. LayerNorm

Parameters: `F1=8`, `D=2` (depth multiplier), `F2=16`, `dropout=0.25`.

**Temporal weight sharing:** When `share_temporal=True`, the temporal conv weights must be the same `nn.Parameter` object across EEGEncoder and MEGEncoder. Implement `share_temporal_weights(other)` explicitly.

### fMRIEncoder (MLP)
1. Linear(n_voxels, 2048) + ReLU + Dropout
2. Linear(2048, 1024) + ReLU + Dropout
3. Linear(1024, embed_dim) + LayerNorm

Log a warning if `n_voxels > 10000` (recommending PCA preprocessing).

### SharedEmbeddingProjector (MLP)
1. Linear(bert_dim, embed_dim) + ReLU
2. Linear(embed_dim, embed_dim) + LayerNorm

### SubjectEmbedding
- Wraps `nn.Embedding(n_subjects, subject_embed_dim)`
- `get_mean_embedding()` returns the mean across all subject embeddings

### EEGDecoder / MEGDecoder (shared _TemporalDecoder base)
1. Concat `shared_emb` + `subject_emb` → (B, embed_dim + subject_embed_dim)
2. Linear → 1024 + ReLU
3. Linear → 2048 + ReLU
4. Reshape → (B, n_channels, 2048 // n_channels)
5. ConvTranspose1d blocks (auto-computed strides from n_timepoints) + BatchNorm + ReLU
6. Final Conv1d(n_channels, n_channels, 1) to mix channels
7. Output: (B, n_channels, n_timepoints)

Number of transposed conv layers and strides must be computed in `__init__` — no hardcoding.

### fMRIDecoder (MLP)
1. Concat → (B, embed_dim + subject_embed_dim)
2. Linear → 1024 + ReLU + Dropout(0.3)
3. Linear → 2048 + ReLU + Dropout(0.3)
4. Linear → n_voxels
No output activation (beta maps can be positive or negative).

---

## Implementation Rules (from AGENTS.md)

- All modules use `nn.Module` properly — no functional-only implementations
- Forward methods include debug-mode shape assertions:
  ```python
  if torch.is_grad_enabled():
      assert x.shape[1] == self.n_channels, f"Expected {self.n_channels} channels, got {x.shape[1]}"
  ```
- Every class has a docstring stating input shape, output shape, and assumptions
- No global state — all configuration passed to `__init__`

---

## Integration Handoff

When done, the integration step copies:
- `worker_models/src/models/` → `src/models/`
- `worker_models/tests/` → `tests/`
