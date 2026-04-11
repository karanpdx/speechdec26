# Spec: Root Train-Ready Pipeline Integration

## Scope

Make the repository root runnable for training by exposing the worker implementations through the canonical project layout:

- `src/data/`
- `src/models/`
- `src/training/`
- `src/evaluation/`
- `scripts/`
- `configs/`

The goal is a train-ready pipeline from processed data onward, with Stage 1 and Stage 2 as separate training runs.

## Inputs

- Worker implementations:
  - `workers/worker_data/src/data/*`
  - `workers/worker_models/src/models/*`
  - `workers/worker_training/src/training/*`
  - `workers/worker_eval/src/evaluation/*`
- Existing worker config files
- Existing worker tests

## Outputs

- Root import paths like `src.models.encoders` and `src.training.train_stage1` work.
- Root scripts:
  - `scripts/train_stage1.py`
  - `scripts/train_stage2.py`
  - optional data helper CLI for `run_alignment`
- Root configs:
  - preprocessing configs
  - splits config
  - train_stage1 config
  - train_stage2 config
- Stage 1 and Stage 2 remain fully separate training entry points:
  - Stage 1 trains encoders/projector
  - Stage 2 requires a Stage 1 checkpoint and freezes Stage 1 encoders

## Constraints

- Do not rewrite worker implementations into the root manually when thin wrappers are sufficient.
- Preserve the worker folders as source-of-truth artifacts for teammates.
- Keep Stage 2 dependent on `stage1_checkpoint`; do not create any code path that trains Stage 2 implicitly after Stage 1.

## Behavior

### Root package wrappers

- Root `src/...` modules should re-export the worker implementations.
- Root package `__init__.py` files should exist for all major subpackages.

### Separate Stage 1 / Stage 2 training

- `scripts/train_stage1.py` calls `src.training.train_stage1.train()`
- `scripts/train_stage2.py` calls `src.training.train_stage2.train()`
- `configs/train_stage2.yaml` must require `stage1_checkpoint`
- Stage 2 should not run without an existing Stage 1 checkpoint

### Train-readiness boundary

- The repo is considered train-ready if:
  - processed data already exists
  - `split_v1.json` exists
  - `vocab_embeddings.npz` exists
  - root Stage 1 and Stage 2 scripts are runnable

- The repo is **not** considered fully raw-to-train ready if the raw-data preprocessors remain unimplemented.

## Success Criteria

- Root imports for `src.data`, `src.models`, `src.training`, and `src.evaluation` resolve.
- Stage 1 and Stage 2 are separate executable scripts from repo root.
- Integrated checks against the training path pass.
- Remaining blockers are limited to raw preprocessing if any.
