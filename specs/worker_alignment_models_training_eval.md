# Spec: Cross-Worker Alignment for Models, Training, and Eval

## Scope

Align `workers/worker_models`, `workers/worker_training`, and `workers/worker_eval` so their exported interfaces, test harnesses, and helper logic are compatible as a single integration unit.

## Inputs

- Model classes from `workers/worker_models/src/models/encoders.py` and `decoders.py`
- Training code from `workers/worker_training/src/training/`
- Evaluation code from `workers/worker_eval/src/evaluation/`
- Existing worker-local tests

## Outputs

- Training code that correctly consumes model-worker interfaces
- Dataset batches carrying the metadata Stage 2 losses require
- Evaluation functions that accept the numpy outputs described in `worker_eval/README.md`
- Worker-local tests runnable from the repo root without import-path failures

## Alignment Fixes

### Training ↔ Models

- `build_models()` must import encoder/projector/subject classes from the actual module that defines them.
- Stage 1 model construction must honor `share_temporal_weights` from config:
  - pass `share_temporal=True` to EEG/MEG encoders when enabled
  - call `eeg_encoder.share_temporal_weights(meg_encoder)` after construction
- Stage 2 training must be able to recover per-modality sampling frequency from dataset batches rather than falling back silently.

### Dataset ↔ Stage 2 Training

- `MultiModalDataset.__getitem__()` must propagate optional metadata used by losses:
  - `sfreq` for EEG/MEG when present in `.npz`
  - `voxel_coords` for fMRI when present in `.npz`
- `collate_fn()` must batch metadata in a form Stage 2 can consume without changing existing Stage 1 behavior.

### Eval ↔ Training Outputs

- Retrieval metrics operate on plain numpy arrays:
  - neural embeddings `(n_samples, embed_dim)`
  - text embeddings `(vocab_size, embed_dim)`
  - labels, vocab, subject IDs
- Reconstruction metrics operate on plain numpy arrays:
  - EEG/MEG epochs `(n_samples, n_channels, n_timepoints)`
  - fMRI beta maps `(n_words, n_voxels)`
  - channel names, voxel coordinates, and sampling frequency
- `worker_eval/src/evaluation/reconstruction.py` must implement:
  - `compute_erp`
  - `compute_n400_amplitude`
  - `compute_n400_correlation`
  - `plot_erp_comparison`
  - `compute_fmri_spatial_correlation`
  - `compute_neurosynth_correlation`
  - `plot_fmri_glass_brain`
  - `compute_round_trip_similarity`
  - `generate_stage2_report`

## Edge Cases

- `share_temporal_weights` disabled: encoders remain independent.
- Missing `sfreq` in `.npz`: Stage 2 can still use config fallback.
- Empty or missing fMRI voxel coordinates: smoothness/visualization helpers degrade gracefully.
- Single-word or constant-valued correlation inputs: return stable zero-correlation results instead of crashing.
- Optional plotting dependencies (`matplotlib`, `nilearn`, `scipy`) missing: fail gracefully with fallback outputs where possible.

## Success Criteria

- `pytest workers/worker_training/tests -q` passes from repo root.
- `pytest workers/worker_models/tests -q` passes from repo root.
- `pytest workers/worker_eval/tests -q` passes from repo root.
- No remaining `NotImplementedError` in the worker areas touched by this request.
