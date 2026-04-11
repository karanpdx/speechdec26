# worker_training — Training Pipeline

## What This Is

The training subsystem for a multimodal neural speech decoding pipeline. Implements all loss functions, the multi-modal dataset loader, and two training loops: Stage 1 (joint contrastive training of EEG/MEG/fMRI encoders) and Stage 2 (decoder training with frozen encoders). This worker is developed in isolation and integrated into the full pipeline once complete.

## Core Value

Loss functions and training loops that produce a stable, non-NaN Stage 1 run — verified by above-chance retrieval accuracy on at least one modality.

## What This Worker Depends On (Interface Contracts)

**From worker_models (Person 2) — use stubs until real models arrive:**
- `EEGEncoder(n_channels, n_timepoints, embed_dim)` → `(B, embed_dim)`
- `MEGEncoder(n_channels, n_timepoints, embed_dim)` → `(B, embed_dim)`
- `fMRIEncoder(n_voxels, embed_dim)` → `(B, embed_dim)`
- `SharedEmbeddingProjector(bert_dim, embed_dim)` → `(B, embed_dim)`
- `SubjectEmbedding(n_subjects, subject_embed_dim)` → `(B, subject_embed_dim)`
- `EEGDecoder / MEGDecoder / fMRIDecoder` → real signal shapes

**From worker_data (Person 1) — use stubs/data_stubs.py until real data arrives:**
- `.npz` schema: `data (float32), labels (list[str]), subject_id (str)`
- `data/splits/split_v1.json`
- `data/processed/vocab_embeddings.npz`

## Requirements

### Validated

(None yet)

### Active

- [ ] `ContrastiveLoss` — CLIP-style symmetric, learnable temperature, clamped, raises on batch_size=1
- [ ] `CrossModalAlignmentLoss` — shared-label mask, returns zero grad for < 2 matched samples
- [ ] `SubjectAdversarialLoss` — gradient reversal, linearly scheduled alpha
- [ ] `MultiModalDataset` — loads `.npz` files, builds balanced batches across modalities
- [ ] `collate_fn` — groups by modality, builds shared_label_mask
- [ ] Stage 1 training loop — joint contrastive, cross-modal, adversarial losses, CSV logging, checkpointing
- [ ] Stage 2 training loop — frozen encoders verified, MSE + freq-domain + spatial-smooth losses
- [ ] `freq_domain_loss` — PSD loss across delta/theta/alpha/beta bands
- [ ] `build_voxel_adjacency` + `spatial_smoothness_loss` for fMRI
- [ ] All loss tests pass (test_losses.py)

### Out of Scope

- Evaluation metrics — that is worker_eval's job
- Model architectures — that is worker_models' job
- Data preprocessing — that is worker_data's job
- Fine-tuning BERT — never

## Constraints

- **Stubs first:** develop and test against `stubs/model_stubs.py` and `stubs/data_stubs.py`; swap at integration
- **Stage 2 gate:** do not implement Stage 2 training until Stage 1 losses are tested and stable
- **Frozen encoders:** assert `not param.requires_grad` for all encoder params before every Stage 2 step
- **No NaN:** contrastive loss temperature must be clamped; validate on synthetic data before real data

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Stubs for development | Eliminates dependency on other workers during implementation | — Pending |
| Losses tested independently | Loss functions are pure tensor ops — no model or data needed | — Pending |
| Stage 2 gate on Stage 1 stability | Training decoders before encoders converge wastes compute | — Pending |

---
*Last updated: 2026-04-10 after initialization*
