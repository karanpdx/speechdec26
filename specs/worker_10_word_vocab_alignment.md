# Spec: Exact 10-Word Shared Vocabulary

## Scope

Update the data alignment and training data loading paths so the project uses exactly 10 words, all 10 are present across EEG, MEG, and fMRI, and null/invalid labels are excluded.

## Inputs

- Processed modality files under `data/processed/<dataset>/<modality>/`
- Split/alignment config in `workers/worker_data/configs/splits.yaml`
- Training dataset loader in `workers/worker_training/src/training/dataset.py`

## Outputs

- A filtered vocabulary of exactly 10 words
- Those 10 words appear in all configured modalities
- `split_v1.json` contains training-compatible filenames, not absolute paths
- Training dataset loader yields only samples whose labels are in the final vocabulary
- Null labels (`None`, `NaN`, empty string, `"null"`, `"none"`, `"nan"`) are excluded

## Behavior

### Alignment

- Build vocabulary from non-null labels only.
- Count frequencies only on training subjects.
- Restrict candidates to words present in all configured modalities.
- Sort candidates by:
  1. descending combined training frequency
  2. alphabetical label for deterministic tie-breaking
- Keep exactly `target_vocab_size=10`.
- Raise `ValueError` if fewer than 10 eligible words exist.

### Split Generation

- Split JSON entries must contain filenames only, matching the interface contract:
  - `sub-01_epochs.npz`
  - `sub-01_betas.npz`
- A file is included in a split only if it contains at least one valid label from the final vocabulary.

### Training Loader

- Loader construction must skip any row whose label is null/invalid or not in the loaded vocabulary.
- This ensures mixed-label files still produce batches using only the final 10 words.

## Edge Cases

- Dataset contains fewer than 10 non-null shared words: fail loudly.
- A modality path is unset: intersection applies only to configured modalities.
- Labels contain whitespace variants: trim before comparison.
- A file contains only invalid/out-of-vocab labels: skip the file’s rows entirely.

## Success Criteria

- `run_alignment()` can produce a 10-word shared vocabulary deterministically.
- `MultiModalDataset` uses only those 10 words at runtime.
- Relevant worker_data and worker_training tests pass for the new behavior.
