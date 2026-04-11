# Roadmap: Multimodal Neural Speech Decoding Pipeline

**Core Value:** A learned shared embedding space that genuinely aligns neural signals across modalities toward BERT word embeddings — verified by above-chance retrieval and positive cross-modal alignment gap.

**Total phases:** 7
**v1 requirements mapped:** 38/38

---

## Phase 1: Data Infrastructure

**Goal:** Every raw modality is preprocessed into validated tensor formats, vocabulary embeddings are ready, and all splits are reproducible with no subject leakage.

**Delivers:**
- `DATASET_CARD.md` for each dataset confirming inspection protocol complete
- EEG tensors `(n_epochs, n_channels, n_timepoints)` with word labels, validation assertions passing
- MEG tensors in identical format after Maxwell/SSS filtering
- fMRI beta maps `(n_words, n_voxels)` via GLM with HRF convolution
- Reproducible train/val/test splits stratified by subject, no leakage
- `data/processed/vocab_embeddings.npz` with BERT embeddings for full shared vocabulary

### Plans

1. **Dataset inspection and DATASET_CARD protocol** — define the inspection checklist and produce `DATASET_CARD.md` for each dataset before any preprocessing code is written; captures sampling rate, channel count, word label format, artifact flags
2. **EEG preprocessing pipeline** — raw recordings to clean epoched tensors with bandpass filter, ICA/autoreject artifact removal, epoch extraction, output validation assertions on shape and label alignment
3. **MEG preprocessing pipeline** — Maxwell/SSS filtering, then identical epoch extraction and validation path as EEG; verify shared output format
4. **fMRI preprocessing pipeline** — GLM with HRF convolution per run, extract per-word beta maps, validate `(n_words, n_voxels)` shape and contrast correctness
5. **Dataset alignment, splits, and vocab embeddings** — align vocabularies across modalities, produce stratified subject-safe train/val/test splits, generate BERT embeddings for full shared vocabulary and save to `data/processed/vocab_embeddings.npz`

---

## Phase 2: Stage 1 Models

**Goal:** All Stage 1 encoder and projector modules are implemented, tested, and documented with shape assertions so the training loop can be assembled against a verified API.

**Delivers:**
- `EEGEncoder` and `MEGEncoder` (EEGNet-style, optional temporal weight sharing)
- `fMRIEncoder` (MLP with layernorm output)
- `SharedEmbeddingProjector` (two-layer MLP + LayerNorm bridging BERT space)
- `SubjectEmbedding` with `get_mean_embedding()` for unseen-subject prior
- Debug-mode shape assertions and input/output shape docstrings on all classes
- `tests/test_encoders.py` passing including temporal weight sharing test

### Plans

1. **EEGEncoder and MEGEncoder** — implement EEGNet-style depthwise + separable conv architecture; wire optional temporal weight sharing via explicit parameter reference; add debug-mode shape assertions and docstrings
2. **fMRIEncoder** — MLP encoder for beta map vectors with configurable hidden dims and layernorm on output; shape assertions and docstring
3. **SharedEmbeddingProjector and SubjectEmbedding** — two-layer MLP + LayerNorm projector from BERT space to shared space; lookup-table subject embedding with `get_mean_embedding()` method
4. **Encoder test suite** — `tests/test_encoders.py` covering forward pass shapes, temporal weight sharing, `get_mean_embedding()`, edge cases (single sample, missing modality)

---

## Phase 3: Stage 1 Training

**Goal:** The Stage 1 training loop jointly trains all modality encoders with contrastive, cross-modal, and adversarial losses, runs stably for at least 5 epochs without NaN, and saves checkpoints with full metric logging.

**Delivers:**
- `ContrastiveLoss` (CLIP-style symmetric, learnable temperature clamped)
- `CrossModalAlignmentLoss` (shared-label masking, graceful zero on empty mask)
- `SubjectAdversarialLoss` (gradient reversal, linearly increasing alpha)
- Training loop handling missing modalities per batch without crashing
- CSV loss log (total, per-modality, adversarial) and checkpoint saving
- Verified NaN-free run for 5+ epochs
- `tests/test_losses.py` passing including edge cases

### Plans

1. **Loss functions** — implement `ContrastiveLoss`, `CrossModalAlignmentLoss`, and `SubjectAdversarialLoss`; temperature clamping, shared-label masking logic, gradient reversal layer
2. **Loss test suite** — `tests/test_losses.py` covering batch size 1, empty mask on cross-modal loss, temperature gradient flow, adversarial reversal sign
3. **Stage 1 training loop** — joint multi-modal training; missing-modality handling per batch; optimizer and LR scheduler setup; gradient clipping
4. **Checkpoint and logging** — CSV logger for total/per-modality/adversarial loss, best-val and periodic checkpoint saving, epoch timing
5. **Stability validation** — run 5 epochs on configured modalities; confirm no NaN loss; smoke test with tiny synthetic data in CI

---

## Phase 4: Stage 1 Evaluation

**Goal:** Retrieval performance is measured end-to-end, cross-subject generalization is quantified per subject, alignment gap is confirmed positive, and a full evaluation report is generated.

**Delivers:**
- `compute_retrieval_metrics`: top-1/5/10 accuracy, MRR via cosine retrieval against vocabulary
- `compute_cross_subject_generalization`: per-subject metrics and mean (not pooled)
- `compute_cross_modal_alignment`: matched vs. random cosine similarity gap (positive gap required)
- `compute_abstention_curve`: coverage/accuracy tradeoff, threshold for 80% accuracy
- `evaluation/stage1_report.md` with all required sections

### Plans

1. **Retrieval metrics** — `compute_retrieval_metrics` function: encode val set, cosine similarity against `vocab_embeddings.npz`, compute top-1/5/10 and MRR; unit test on synthetic embeddings
2. **Cross-subject generalization** — `compute_cross_subject_generalization`: per-subject loop, aggregate mean ± std; verify no pooling across subjects
3. **Cross-modal alignment and abstention** — `compute_cross_modal_alignment` (matched vs. random gap); `compute_abstention_curve` with threshold search for 80% accuracy; visualizations
4. **Stage 1 evaluation report** — run all metrics on best checkpoint, write `evaluation/stage1_report.md` with retrieval table, per-subject breakdown, alignment gap, abstention plot paths, and interpretation notes

---

## Phase 5: Stage 2 Models

**Goal:** All Stage 2 decoder modules are implemented and tested, capable of reconstructing neural signals from shared + subject embeddings.

**Delivers:**
- `_TemporalDecoder` base class with auto-computed transposed conv layers
- `EEGDecoder` and `MEGDecoder` reconstructing `(batch, n_channels, n_timepoints)`
- `fMRIDecoder` reconstructing `(batch, n_voxels)` beta maps with no output activation
- `tests/test_decoders.py` passing

### Plans

1. **TemporalDecoder base and EEGDecoder/MEGDecoder** — `_TemporalDecoder` with `n_timepoints`-driven automatic transposed conv depth; `EEGDecoder` and `MEGDecoder` subclasses accepting shared + subject embeddings; shape assertions and docstrings
2. **fMRIDecoder** — MLP decoder to `(batch, n_voxels)` with no output activation; configurable hidden dims; shape assertions
3. **Decoder test suite** — `tests/test_decoders.py` covering output shape, no-activation check on fMRI output, round-trip dimensionality, gradient flow through subject embedding

---

## Phase 6: Stage 2 Training

**Goal:** Stage 2 trains with strictly frozen Stage 1 encoders, combining MSE, frequency-domain PSD, and spatial smoothness losses, and achieves decreasing reconstruction MSE over training.

**Delivers:**
- Verified frozen encoder weights before first gradient step
- Frequency-domain loss penalizing PSD differences across delta/theta/alpha/beta bands
- Spatial smoothness loss on fMRI predictions using precomputed voxel adjacency
- Confirmed MSE decrease over training (not stuck at initialization)

### Plans

1. **Frozen encoder loading and verification** — load Stage 1 checkpoint, freeze all encoder parameters, assert no encoder gradients before training starts; fail loudly if verification fails
2. **Frequency-domain loss** — PSD loss over delta/theta/alpha/beta bands for EEG/MEG; Welch or FFT-based implementation; unit test against known synthetic signal
3. **Spatial smoothness loss** — precompute voxel adjacency from coordinates, penalize adjacent-voxel prediction differences; unit test on small synthetic beta map
4. **Stage 2 training loop** — combine MSE + frequency-domain + spatial-smoothness with configurable weights; checkpoint saving; confirm MSE curve decreases over epochs on held-out validation set

---

## Phase 7: Stage 2 Evaluation and Integration

**Goal:** Stage 2 reconstruction is validated on neurophysiologically meaningful criteria (ERP structure, fMRI spatial patterns, round-trip consistency), and the full integration checklist passes end-to-end.

**Delivers:**
- N400 amplitude Pearson r between real and predicted waveforms at centro-parietal channels
- fMRI spatial correlation mean ± std across words and subjects
- Neurosynth language map correlation reported
- Round-trip cosine similarity > 0.5 (BERT → shared → decoder → encoder → shared)
- `evaluation/stage2_report.md`
- Full integration checklist in AGENTS.md passing

### Plans

1. **ERP component recovery** — extract N400 window at centro-parietal channels from real and predicted EEG waveforms; compute Pearson r; plot overlay
2. **fMRI spatial evaluation** — Pearson r between predicted and real beta maps per word per subject, aggregate mean ± std; Neurosynth language map correlation
3. **Round-trip consistency** — BERT embed word → shared space → decoder → re-encode → shared space; compute cosine similarity; assert mean > 0.5 threshold
4. **Stage 2 evaluation report** — compile all metrics into `evaluation/stage2_report.md` with ERP plots, spatial correlation table, round-trip histogram, and interpretation
5. **Full integration checklist** — run end-to-end pipeline from raw data through both stages; verify all assertions in AGENTS.md integration checklist pass; document any deviations

---

## Requirements Coverage

| Phase | Requirements |
|-------|-------------|
| Phase 1: Data Infrastructure | DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06 |
| Phase 2: Stage 1 Models | S1M-01, S1M-02, S1M-03, S1M-04, S1M-05, S1M-06, TEST-01 |
| Phase 3: Stage 1 Training | S1T-01, S1T-02, S1T-03, S1T-04, S1T-05, S1T-06, TEST-02 |
| Phase 4: Stage 1 Evaluation | S1E-01, S1E-02, S1E-03, S1E-04, S1E-05 |
| Phase 5: Stage 2 Models | S2M-01, S2M-02, S2M-03, TEST-03 |
| Phase 6: Stage 2 Training | S2T-01, S2T-02, S2T-03, S2T-04 |
| Phase 7: Stage 2 Evaluation + Integration | S2E-01, S2E-02, S2E-03, S2E-04, S2E-05, TEST-04 |

**Total v1 requirements:** 38
**Mapped:** 38
**Unmapped:** 0

---
*Roadmap created: 2026-04-10*
