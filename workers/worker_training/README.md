# Worker Training — Person 3

## Your Job

Implement all loss functions, the multi-modal dataset loader, and both training loops (Stage 1 and Stage 2). You train the full pipeline.

**You depend on Person 2's model interfaces and Person 1's data schemas.** Use the stubs in `stubs/` while developing — they return correctly-shaped tensors without any real logic.

---

## What You Build

```
worker_training/
├── src/training/
│   ├── losses.py       # ContrastiveLoss, CrossModalAlignmentLoss, SubjectAdversarialLoss
│   ├── dataset.py      # MultiModalDataset — loads .npz files, builds batches
│   ├── train_stage1.py # Joint multi-modal contrastive training loop
│   └── train_stage2.py # Stage 2 decoder training (frozen Stage 1 encoders)
├── scripts/
│   ├── train_stage1.py # CLI entry point
│   └── train_stage2.py # CLI entry point
├── configs/
│   ├── train_stage1.yaml
│   └── train_stage2.yaml
├── stubs/
│   ├── model_stubs.py  # Fake encoders/decoders returning correct shapes
│   └── data_stubs.py   # Fake dataset returning correct .npz schemas
└── tests/
    └── test_losses.py
```

---

## Interface Contract — What You Import

### From Person 2 (worker_models → src/models)
```python
from src.models.encoders import (
    EEGEncoder,              # forward(x: (B, C, T)) → (B, embed_dim)
    MEGEncoder,              # forward(x: (B, C, T)) → (B, embed_dim)
    fMRIEncoder,             # forward(x: (B, V)) → (B, embed_dim)
    SharedEmbeddingProjector,# forward(x: (B, 768)) → (B, embed_dim)
    SubjectEmbedding,        # forward(ids: (B,)) → (B, subject_embed_dim)
)
from src.models.decoders import EEGDecoder, MEGDecoder, fMRIDecoder
```

**During development:** use `stubs/model_stubs.py` which provides drop-in replacements.

### From Person 1 (data files)
```
data/processed/<dataset>/eeg/sub-<id>_epochs.npz
    .data:           float32 (n_epochs, n_channels, n_timepoints)
    .labels:         list[str]  length n_epochs
    .subject_id:     str
    .sfreq:          float
    .ch_names:       list[str]
    .event_onsets_s: float64 (n_epochs,)

data/processed/<dataset>/fmri/sub-<id>_betas.npz
    .data:           float32 (n_words, n_voxels)
    .labels:         list[str]  length n_words
    .subject_id:     str
    .voxel_coords:   int32 (n_voxels, 3)
    .pca_explained_variance: float

data/processed/vocab_embeddings.npz
    .vocab:          list[str]  length V
    .embeddings:     float32 (V, 768)

data/splits/split_v1.json
    {"train": {"eeg": [...], "meg": [...], "fmri": [...]}, "val": {...}, "test": {...}}
```

**During development:** use `stubs/data_stubs.py` to generate synthetic equivalents.

---

## Stage 1 Training Logic

For each batch:
1. Forward pass each modality encoder for samples of that modality
2. Forward pass `SharedEmbeddingProjector` for corresponding BERT embeddings
3. `ContrastiveLoss` per modality — sum them
4. `CrossModalAlignmentLoss` for any modality pairs sharing stimulus labels in the batch
5. `SubjectAdversarialLoss` on all shared embeddings — scale by alpha (linear 0→1 over training)
6. Total loss = contrastive + λ_cross * cross_modal + λ_adv * adversarial
7. Backward, gradient clip (max_norm=1.0), optimizer step

**Handle missing modalities:** if a batch contains only EEG (no MEG, no fMRI), compute only the EEG contrastive loss. Do not crash on missing modalities.

**Validation:** every `val_every_n_epochs`, run retrieval evaluation (top-1, top-5 accuracy) on the val split. Log per modality.

---

## Stage 2 Training Logic

1. Load Stage 1 checkpoint and freeze ALL encoder weights
2. Verify frozen: `assert not param.requires_grad for param in encoder.parameters()`
3. For each batch: forward through frozen encoder → shared embedding → decoder → reconstructed signal
4. Loss: MSE (primary) + frequency-domain PSD loss (EEG/MEG) + spatial smoothness (fMRI)
5. Backward on decoder parameters only

---

## Critical: Do Not Cross This Line

- Do not train Stage 2 before Stage 1 has converged
- Do not fine-tune BERT (it is never loaded in training — only embeddings are used)
- Do not report training-split accuracy as a performance metric
- Stage 2 encoder weights must be frozen — verify with assertions, not comments

---

## Development Workflow

1. Write and test losses first (`losses.py` + `tests/test_losses.py`) — no models needed
2. Write and test `dataset.py` using `stubs/data_stubs.py`
3. Write `train_stage1.py` using `stubs/model_stubs.py` + `stubs/data_stubs.py`
4. Write `train_stage2.py` the same way
5. At integration, swap stubs for real models/data

---

## Integration Handoff

When done, the integration step copies:
- `worker_training/src/training/` → `src/training/`
- `worker_training/scripts/` → `scripts/`
- `worker_training/configs/` → `configs/`
- `worker_training/tests/` → `tests/`
