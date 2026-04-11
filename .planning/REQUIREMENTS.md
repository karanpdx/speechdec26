# Requirements: Multimodal Neural Speech Decoding Pipeline

**Defined:** 2026-04-10
**Core Value:** A learned shared embedding space that genuinely aligns neural signals across modalities toward BERT word embeddings — verified by above-chance retrieval and positive cross-modal alignment gap.

## v1 Requirements

### Data Infrastructure

- [ ] **DATA-01**: Dataset inspection protocol runs for every new dataset before any preprocessing code is written, producing a `DATASET_CARD.md`
- [ ] **DATA-02**: EEG preprocessing pipeline transforms raw recordings into clean epoched tensors `(n_epochs, n_channels, n_timepoints)` with word labels and passes output validation assertions
- [ ] **DATA-03**: MEG preprocessing pipeline applies Maxwell/SSS filtering and produces identical output format to EEG pipeline
- [ ] **DATA-04**: fMRI preprocessing pipeline produces per-word beta maps `(n_words, n_voxels)` via a GLM with HRF convolution and passes output validation assertions
- [ ] **DATA-05**: Dataset alignment step produces consistent vocabulary across modalities and a reproducible train/val/test split with no subject leakage
- [ ] **DATA-06**: BERT vocabulary embeddings generated for the full shared vocabulary and saved to `data/processed/vocab_embeddings.npz`

### Stage 1 — Models

- [ ] **S1M-01**: `EEGEncoder` implements EEGNet-style architecture (depthwise + separable conv) with configurable channels, timepoints, and embed_dim
- [ ] **S1M-02**: `MEGEncoder` shares architecture with EEGEncoder and supports optional temporal weight sharing via explicit parameter reference
- [ ] **S1M-03**: `fMRIEncoder` implements MLP encoder for beta map vectors with layernorm output
- [ ] **S1M-04**: `SharedEmbeddingProjector` bridges BERT space to learned shared space (two-layer MLP + LayerNorm)
- [ ] **S1M-05**: `SubjectEmbedding` lookup table with `get_mean_embedding()` method for unseen-subject prior
- [ ] **S1M-06**: All encoder forward methods include debug-mode shape assertions; all classes have input/output shape docstrings

### Stage 1 — Training

- [ ] **S1T-01**: `ContrastiveLoss` implements CLIP-style symmetric contrastive loss with learnable temperature clamped to `[log(0.01), log(100)]`
- [ ] **S1T-02**: `CrossModalAlignmentLoss` computes loss only over samples with a shared label in both modalities; returns zero gracefully for empty masks
- [ ] **S1T-03**: `SubjectAdversarialLoss` implements gradient reversal with linearly increasing alpha; small classifier inside the loss module
- [ ] **S1T-04**: Stage 1 training loop jointly trains all modality encoders; handles missing modalities per batch without crashing
- [ ] **S1T-05**: Training loop logs total loss, per-modality losses, and adversarial loss to CSV; saves best val checkpoint and periodic checkpoints
- [ ] **S1T-06**: Training runs without NaN loss for at least 5 epochs on the configured modalities

### Stage 1 — Evaluation

- [ ] **S1E-01**: `compute_retrieval_metrics` returns top-1, top-5, top-10 accuracy and MRR via cosine retrieval against vocabulary
- [ ] **S1E-02**: `compute_cross_subject_generalization` reports per-subject metrics and mean across subjects (not pooled)
- [ ] **S1E-03**: `compute_cross_modal_alignment` reports matched vs. random cosine similarity gap (must be positive)
- [ ] **S1E-04**: `compute_abstention_curve` returns coverage/accuracy tradeoff; identifies threshold for 80% accuracy at max coverage
- [ ] **S1E-05**: Stage 1 evaluation report generated at `evaluation/stage1_report.md` with all required sections

### Stage 2 — Models

- [ ] **S2M-01**: `_TemporalDecoder` base class shared by `EEGDecoder` and `MEGDecoder`; number of transposed conv layers computed automatically from `n_timepoints`
- [ ] **S2M-02**: `EEGDecoder` and `MEGDecoder` reconstruct `(batch, n_channels, n_timepoints)` from shared + subject embeddings
- [ ] **S2M-03**: `fMRIDecoder` reconstructs `(batch, n_voxels)` beta maps via MLP; no output activation

### Stage 2 — Training

- [ ] **S2T-01**: Stage 2 training loads Stage 1 checkpoint and verifies all encoder weights are frozen before any gradient step
- [ ] **S2T-02**: Frequency-domain loss penalizes PSD differences across delta/theta/alpha/beta bands for EEG/MEG reconstruction
- [ ] **S2T-03**: Spatial smoothness loss penalizes adjacent-voxel differences in predicted fMRI beta maps using precomputed adjacency from voxel coords
- [ ] **S2T-04**: Reconstruction MSE decreases over training (not stuck at initialization)

### Stage 2 — Evaluation

- [ ] **S2E-01**: ERP component recovery: N400 amplitude correlation (Pearson r) between real and predicted waveforms at centro-parietal channels
- [ ] **S2E-02**: fMRI spatial correlation: Pearson r between predicted and real beta maps across voxels, reported as mean ± std across words and subjects
- [ ] **S2E-03**: Neurosynth language map correlation reported as evidence of spatial pattern recovery
- [ ] **S2E-04**: Round-trip cosine similarity > 0.5: BERT → shared space → decoder → encoder → shared space cosine similarity
- [ ] **S2E-05**: Stage 2 evaluation report generated at `evaluation/stage2_report.md`

### Testing and Integration

- [ ] **TEST-01**: All encoder tests pass (`tests/test_encoders.py`) including temporal weight sharing test
- [ ] **TEST-02**: All loss function tests pass (`tests/test_losses.py`) including edge cases (batch size 1, empty mask)
- [ ] **TEST-03**: All decoder tests pass (`tests/test_decoders.py`)
- [ ] **TEST-04**: Full integration checklist in AGENTS.md passes end-to-end

## v2 Requirements

### Extensions

- **EXT-01**: Live EEG domain shift correction via Euclidean alignment at inference
- **EXT-02**: Sentence-level decoding (multi-word targets, not single-word retrieval)
- **EXT-03**: Cross-dataset subject transfer (train on dataset A, evaluate on dataset B subjects)
- **EXT-04**: Streaming / real-time inference pipeline

### Usability

- **USE-01**: CLI entry points with argument parsing for all preprocessing and training scripts
- **USE-02**: Weights & Biases or MLflow integration for experiment tracking

## Out of Scope

| Feature | Reason |
|---------|--------|
| BERT fine-tuning | BERT is frozen supervision throughout; modifying it would destabilize the contrastive targets |
| OOV generalization | Model is not trained on OOV words; failure is expected and documented, not a bug to fix |
| Mobile / real-time inference | Offline batch research pipeline only in v1 |
| Cross-dataset normalization | Dataset-specific preprocessing; cross-dataset alignment is a research problem deferred to v2 |
| Subject-independent model (zero-shot) | Requires held-out subject protocol beyond current scope; v1 uses held-out subjects for eval only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Pending |
| DATA-06 | Phase 1 | Pending |
| S1M-01 | Phase 2 | Pending |
| S1M-02 | Phase 2 | Pending |
| S1M-03 | Phase 2 | Pending |
| S1M-04 | Phase 2 | Pending |
| S1M-05 | Phase 2 | Pending |
| S1M-06 | Phase 2 | Pending |
| S1T-01 | Phase 3 | Pending |
| S1T-02 | Phase 3 | Pending |
| S1T-03 | Phase 3 | Pending |
| S1T-04 | Phase 3 | Pending |
| S1T-05 | Phase 3 | Pending |
| S1T-06 | Phase 3 | Pending |
| S1E-01 | Phase 4 | Pending |
| S1E-02 | Phase 4 | Pending |
| S1E-03 | Phase 4 | Pending |
| S1E-04 | Phase 4 | Pending |
| S1E-05 | Phase 4 | Pending |
| S2M-01 | Phase 5 | Pending |
| S2M-02 | Phase 5 | Pending |
| S2M-03 | Phase 5 | Pending |
| S2T-01 | Phase 6 | Pending |
| S2T-02 | Phase 6 | Pending |
| S2T-03 | Phase 6 | Pending |
| S2T-04 | Phase 6 | Pending |
| S2E-01 | Phase 7 | Pending |
| S2E-02 | Phase 7 | Pending |
| S2E-03 | Phase 7 | Pending |
| S2E-04 | Phase 7 | Pending |
| S2E-05 | Phase 7 | Pending |
| TEST-01 | Phase 2 | Pending |
| TEST-02 | Phase 3 | Pending |
| TEST-03 | Phase 5 | Pending |
| TEST-04 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 38 total
- Mapped to phases: 38
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-10*
*Last updated: 2026-04-10 after initial definition*
