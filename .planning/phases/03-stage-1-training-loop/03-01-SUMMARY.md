---
phase: 03-stage-1-training-loop
plan: "01"
subsystem: worker_training
tags: [training, models, optimizer, stage1]
dependency_graph:
  requires: []
  provides: [build_models, build_optimizer, compute_alpha]
  affects: [train_one_epoch, validate, save_checkpoint]
tech_stack:
  added: []
  patterns: [AdamW optimizer, ImportError stub fallback, gradient reversal]
key_files:
  created: []
  modified:
    - workers/worker_training/src/training/train_stage1.py
    - workers/worker_training/src/training/losses.py
    - workers/worker_training/configs/train_stage1.yaml
decisions:
  - "Used config.get('n_subjects') or 20 pattern to handle both null yaml value and missing key"
  - "Included adversarial_loss parameters in optimizer — its internal classifier must be trained jointly"
metrics:
  duration_minutes: 10
  completed: 2026-04-10
  tasks_completed: 2
  files_modified: 3
---

# Phase 03 Plan 01: Model Instantiation and Optimizer Setup Summary

**One-liner:** AdamW optimizer over six Stage 1 model components with ImportError-based stub fallback and linear alpha schedule for gradient reversal.

## What Was Built

Three functions in `workers/worker_training/src/training/train_stage1.py`:

- **build_models**: Tries to import from `src.models`; falls back to `stubs/model_stubs.py` with a logged warning. Instantiates EEGEncoder, MEGEncoder, fMRIEncoder, SharedEmbeddingProjector, SubjectEmbedding, and SubjectAdversarialLoss from config keys. Moves all models to the configured device.

- **build_optimizer**: Collects parameters from all six model components (including the adversarial classifier) into a single AdamW optimizer with lr=3e-4 and weight_decay=1e-4 from config.

- **compute_alpha**: Returns `current_epoch / (total_epochs - 1)` — exactly 0.0 at epoch 0 and 1.0 at the final epoch. Handles the edge case of total_epochs <= 1 by returning 1.0.

`configs/train_stage1.yaml`: `n_subjects` changed from `null` to `20` so the default config runs without manual editing.

## Verification Results

```
PASS: build_models and compute_alpha
PASS: build_optimizer with 14 param tensors
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] SubjectAdversarialLoss was a NotImplementedError stub in losses.py**

- **Found during:** Task 1 verification
- **Issue:** `build_models` calls `SubjectAdversarialLoss(embed_dim, n_subjects)` from `src.training.losses`, but the `__init__` raised `NotImplementedError`, preventing instantiation entirely.
- **Fix:** Implemented `SubjectAdversarialLoss.__init__` (two-layer MLP classifier) and `forward` (gradient reversal via `GradientReversalFunction.apply` + cross-entropy). This matches the documented interface exactly.
- **Files modified:** `workers/worker_training/src/training/losses.py`
- **Commit:** 631e2b4

## Commits

| Hash    | Message                                                                         |
|---------|---------------------------------------------------------------------------------|
| 631e2b4 | feat(03-01): implement build_models, compute_alpha, and SubjectAdversarialLoss  |
| f613b3b | feat(03-01): implement build_optimizer; set n_subjects default to 20            |

## Known Stubs

The following functions in `train_stage1.py` remain as `NotImplementedError` stubs — intentional per plan scope:

- `train_one_epoch` — to be implemented in plan 03-02
- `validate` — to be implemented in a later plan
- `save_checkpoint` — to be implemented in a later plan
- `train` (entry point) — to be implemented in a later plan

`ContrastiveLoss` and `CrossModalAlignmentLoss` in `losses.py` also remain as stubs — these are addressed in a separate losses plan.

## Self-Check: PASSED

- `workers/worker_training/src/training/train_stage1.py` — modified, exists
- `workers/worker_training/src/training/losses.py` — modified, exists
- `workers/worker_training/configs/train_stage1.yaml` — modified, exists
- Commit 631e2b4 — present in git log
- Commit f613b3b — present in git log
