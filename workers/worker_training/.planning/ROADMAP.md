# Roadmap: worker_training

**4 phases. All work is inside `workers/worker_training/`.**
**Develop with stubs. Swap at integration.**

---

## Phase 1: Loss Functions

**Goal:** All three loss functions implemented, tested, and passing.

**Delivers:**
- `src/training/losses.py` — `ContrastiveLoss`, `CrossModalAlignmentLoss`, `SubjectAdversarialLoss`
- `tests/test_losses.py` — all tests passing (including from AGENTS.md verbatim)

### Plans

1. **Implement ContrastiveLoss** — CLIP-style symmetric cross-entropy, learnable temperature, clamp to `[log(0.01), log(100)]`, L2-normalize inputs internally, raise ValueError for batch_size=1
2. **Implement CrossModalAlignmentLoss** — shared-label mask filtering, graceful zero return for < 2 matches, contrastive loss on masked subset
3. **Implement SubjectAdversarialLoss** — GradientReversalFunction (custom autograd), internal subject classifier (embed_dim → hidden → n_subjects), alpha scaling
4. **Verify all loss tests pass** — run `pytest tests/test_losses.py -v` and confirm all assertions green, including gradient reversal and temperature clamping tests

**Requirements covered:** LOSS-01 through LOSS-09

---

## Phase 2: Dataset Loader

**Goal:** `MultiModalDataset` loads stub data end-to-end and produces correctly-shaped batches.

**Delivers:**
- `src/training/dataset.py` — `MultiModalDataset`, `collate_fn`, `build_shared_label_mask`
- Tested against `stubs/data_stubs.py` synthetic `.npz` files

### Plans

1. **Implement MultiModalDataset** — loads `.npz` files from split JSON, maps labels to vocab indices and BERT embeddings, handles missing modalities gracefully
2. **Implement collate_fn** — groups samples by modality, stacks tensors, produces per-modality sub-batches
3. **Implement build_shared_label_mask** — identifies samples with cross-modality word label matches within a batch
4. **End-to-end dataset test** — use `stubs/data_stubs.py` to write a synthetic dataset, load it with `MultiModalDataset`, iterate 3 batches, assert shapes are correct

**Requirements covered:** DATA-01 through DATA-05

---

## Phase 3: Stage 1 Training Loop

**Goal:** Full Stage 1 training loop runs 5 epochs on stub data without NaN.

**Delivers:**
- `src/training/train_stage1.py` — complete training loop, optimizer, scheduler, CSV logger, checkpointing
- `scripts/train_stage1.py` — CLI entry point
- `configs/train_stage1.yaml` — filled in with sensible defaults

### Plans

1. **Implement build_models and build_optimizer** — instantiate encoders/projector/subject_emb/adversarial_loss from config; fall back to model stubs if `src.models` not importable
2. **Implement train_one_epoch** — forward pass per present modality, loss computation, gradient clip, optimizer step, alpha schedule, CSV logging
3. **Implement validate and save_checkpoint** — retrieval accuracy on val split (stub eval), best/periodic checkpoint saving
4. **Smoke test: 5 epochs no NaN** — run training on stub data using `stubs/data_stubs.py` + `stubs/model_stubs.py`; assert no NaN in any loss; verify CSV log written; verify checkpoint saved

**Requirements covered:** S1-01 through S1-07

---

## Phase 4: Stage 2 Training Loop

**Goal:** Stage 2 training loop runs with frozen encoders, MSE decreases, all three loss components work.

**Delivers:**
- `src/training/train_stage2.py` — complete Stage 2 loop with frozen-encoder verification
- `scripts/train_stage2.py` — CLI entry point
- `configs/train_stage2.yaml` — filled in

### Plans

1. **Implement load_stage1_checkpoint and verify_encoders_frozen** — load checkpoint, set `requires_grad=False` on all encoder params, assert before any forward pass
2. **Implement freq_domain_loss** — FFT on time dimension, band-specific PSD MSE across delta/theta/alpha/beta
3. **Implement build_voxel_adjacency and spatial_smoothness_loss** — precompute adjacency from MNI coords at startup, per-batch smoothness penalty on adjacent voxels
4. **Implement train_one_epoch and train** — decoder-only optimizer, combined loss (MSE + freq + spatial), verify MSE decreases over 5 epochs on stub data
5. **Frozen encoder assertion test** — manually set a decoder checkpoint, verify assertions fire if any encoder param has requires_grad=True

**Requirements covered:** S2-01 through S2-07

---

## Requirements Coverage

| Phase | Requirements |
|-------|-------------|
| Phase 1 | LOSS-01 to LOSS-09 |
| Phase 2 | DATA-01 to DATA-05 |
| Phase 3 | S1-01 to S1-07 |
| Phase 4 | S2-01 to S2-07 |

**Total v1:** 26 requirements, 26 mapped ✓
