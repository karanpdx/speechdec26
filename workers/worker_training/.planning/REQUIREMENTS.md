# Requirements: worker_training

**Defined:** 2026-04-10
**Core Value:** Stable, tested training loops that produce non-NaN Stage 1 loss and above-chance retrieval on at least one modality.

## v1 Requirements

### Loss Functions

- [ ] **LOSS-01**: `ContrastiveLoss` computes CLIP-style symmetric cross-entropy with learnable temperature
- [ ] **LOSS-02**: `ContrastiveLoss` temperature is clamped to `[log(0.01), log(100)]` to prevent instability
- [ ] **LOSS-03**: `ContrastiveLoss` raises `ValueError` for batch size 1
- [ ] **LOSS-04**: `ContrastiveLoss` L2-normalizes inputs internally (caller need not pre-normalize)
- [ ] **LOSS-05**: `CrossModalAlignmentLoss` computes loss only over samples where `shared_label_mask` is True
- [ ] **LOSS-06**: `CrossModalAlignmentLoss` returns `tensor(0.0, requires_grad=True)` when < 2 matched samples
- [ ] **LOSS-07**: `SubjectAdversarialLoss` implements `GradientReversalFunction` with `alpha` scaling
- [ ] **LOSS-08**: `SubjectAdversarialLoss` contains a small internal subject classifier
- [ ] **LOSS-09**: All loss tests in `tests/test_losses.py` pass

### Dataset Loader

- [ ] **DATA-01**: `MultiModalDataset` loads `.npz` files from all specified modalities
- [ ] **DATA-02**: `MultiModalDataset` maps word labels to vocabulary indices and BERT embeddings
- [ ] **DATA-03**: `collate_fn` groups samples by modality and stacks tensors
- [ ] **DATA-04**: `build_shared_label_mask` correctly identifies samples with cross-modality label matches
- [ ] **DATA-05**: Dataset works with `stubs/data_stubs.py` synthetic data end-to-end

### Stage 1 Training

- [ ] **S1-01**: Training loop handles missing modalities per batch without crashing
- [ ] **S1-02**: Training loop computes total loss = contrastive + λ_cross * cross_modal + λ_adv * adversarial
- [ ] **S1-03**: Alpha increases linearly from 0 to 1 over `n_epochs`
- [ ] **S1-04**: Gradient clipping applied (max_norm=1.0) before optimizer step
- [ ] **S1-05**: Training logs to CSV: epoch, step, total_loss, per-modality losses, adversarial loss
- [ ] **S1-06**: Best val checkpoint and periodic (every 10 epochs) checkpoints saved
- [ ] **S1-07**: Training runs for 5 epochs without NaN loss on stub data

### Stage 2 Training

- [ ] **S2-01**: Stage 2 loads Stage 1 checkpoint and freezes all encoder weights before any gradient step
- [ ] **S2-02**: `verify_encoders_frozen()` called before every training step; raises `AssertionError` if violated
- [ ] **S2-03**: `freq_domain_loss` penalizes PSD differences across delta/theta/alpha/beta bands
- [ ] **S2-04**: `build_voxel_adjacency` computes adjacency from MNI coords at startup (not per batch)
- [ ] **S2-05**: `spatial_smoothness_loss` uses precomputed adjacency indices
- [ ] **S2-06**: Stage 2 optimizer covers decoder parameters only (not encoder params)
- [ ] **S2-07**: Reconstruction MSE decreases over training on stub data

## Out of Scope

| Feature | Reason |
|---------|--------|
| Evaluation metrics | worker_eval's responsibility |
| Model architectures | worker_models' responsibility |
| Data preprocessing | worker_data's responsibility |
| BERT loading or fine-tuning | BERT embeddings are pre-computed by worker_data |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| LOSS-01 through LOSS-09 | Phase 1 | Pending |
| DATA-01 through DATA-05 | Phase 2 | Pending |
| S1-01 through S1-07     | Phase 3 | Pending |
| S2-01 through S2-07     | Phase 4 | Pending |

**Coverage:** 26 v1 requirements, all mapped ✓

---
*Requirements defined: 2026-04-10*
