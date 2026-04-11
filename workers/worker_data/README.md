# Worker Data — Person 1

## Your Job

Build the entire data pipeline: inspect datasets, preprocess EEG/MEG/fMRI signals, align vocabularies across modalities, generate train/val/test splits, and produce BERT vocabulary embeddings.

**You are the foundation.** Every other worker depends on the file schemas you produce. Do not change the output schemas — they are the interface contract.

---

## What You Build

```
worker_data/
├── src/data/
│   ├── inspect.py          # Dataset inspection + DATASET_CARD.md generation
│   ├── preprocess_eeg.py   # Raw EEG → (n_epochs, n_channels, n_timepoints) .npz
│   ├── preprocess_meg.py   # Raw MEG → same format, Maxwell/SSS filtering
│   ├── preprocess_fmri.py  # Raw fMRI BOLD → (n_words, n_voxels) beta map .npz
│   ├── align_splits.py     # Vocabulary alignment + train/val/test split generation
│   └── vocab_embeddings.py # BERT embeddings for full vocabulary
├── configs/
│   ├── preprocessing_eeg.yaml
│   ├── preprocessing_meg.yaml
│   ├── preprocessing_fmri.yaml
│   └── splits.yaml
└── tests/
    └── test_preprocessing.py
```

---

## Output Files (Interface Contract — Do Not Change Schemas)

### EEG / MEG epochs
**Path:** `data/processed/<dataset_name>/eeg/sub-<id>_epochs.npz`
**Path:** `data/processed/<dataset_name>/meg/sub-<id>_epochs.npz`

```python
{
    "data":          np.float32,  # shape (n_epochs, n_channels, n_timepoints)
    "labels":        list[str],   # length n_epochs, word strings
    "subject_id":    str,
    "sfreq":         float,
    "ch_names":      list[str],
    "event_onsets_s": np.float64, # shape (n_epochs,)
}
```

### fMRI beta maps
**Path:** `data/processed/<dataset_name>/fmri/sub-<id>_betas.npz`

```python
{
    "data":                   np.float32,  # shape (n_words, n_voxels)
    "labels":                 list[str],   # length n_words
    "subject_id":             str,
    "voxel_coords":           np.int32,    # shape (n_voxels, 3) MNI coords
    "pca_explained_variance": float,       # only if PCA applied, else 0.0
}
```

### Vocabulary embeddings
**Path:** `data/processed/vocab_embeddings.npz`

```python
{
    "vocab":       list[str],   # length V, word strings
    "embeddings":  np.float32,  # shape (V, 768)
}
```

### Split index
**Path:** `data/splits/split_v1.json`

```json
{
  "train": {"eeg": ["sub-01_epochs.npz", ...], "meg": [...], "fmri": [...]},
  "val":   {"eeg": [...], "meg": [...], "fmri": [...]},
  "test":  {"eeg": [...], "meg": [...], "fmri": [...]}
}
```

---

## Output Validation

Every output `.npz` must pass these assertions before you consider a module done:

**EEG/MEG:**
```python
assert data.dtype == np.float32
assert data.ndim == 3                  # (n_epochs, n_channels, n_timepoints)
assert len(labels) == data.shape[0]
assert not np.any(np.isnan(data))
assert not np.any(np.isinf(data))
assert abs(data.mean()) < 0.1          # approximately z-scored
assert abs(data.std() - 1.0) < 0.2
```

**fMRI:**
```python
assert data.dtype == np.float32
assert data.ndim == 2                  # (n_words, n_voxels)
assert len(labels) == data.shape[0]
assert not np.any(np.isnan(data))
assert data.std() > 0                  # not all zeros
```

---

## Dataset Inspection Protocol

**Before writing any preprocessing code for a new dataset**, run the inspection protocol:

1. `inspect.py audit_dataset(root_path)` — directory tree, file types, size
2. Identify format (`.fif`, `.edf`, `.set`, `.mat`, `.npy`, `.nii`, BIDS, HDF5)
3. Inspect signal properties (sfreq, channels, duration)
4. Inspect stimulus/label structure — word-level vs sentence-level
5. Document subject/session structure
6. Write `data/processed/<dataset_name>/DATASET_CARD.md`

**Do not write preprocessing code until the dataset card is complete.**

---

## Development Without Real Data

You can develop and test your preprocessors without real datasets by generating synthetic data that matches the expected raw formats:

```python
# Synthetic EEG-like MNE raw for testing
import mne
import numpy as np

sfreq = 256
n_channels = 64
n_seconds = 300
ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_data = np.random.randn(n_channels, int(sfreq * n_seconds)) * 1e-6
raw = mne.io.RawArray(raw_data, info)
```

All tests in `tests/test_preprocessing.py` must use synthetic data — no real datasets required.

---

## Rules (from AGENTS.md)

- Never modify files in `data/raw/` — read only
- Log tensor shapes at every step using Python `logging` at INFO level (not print)
- If labels are sentence-level rather than word-level, raise a clear error — do not silently assume
- If a subject has fewer than 10 epochs, log a warning and skip — do not include in output
- All config is passed via YAML files in `configs/` — no hardcoded paths

---

## Integration Handoff

When your work is done, the integration step will copy:
- `worker_data/src/data/` → `src/data/`
- `worker_data/configs/` → `configs/`
- `worker_data/tests/` → `tests/`

The processed data files in `data/processed/` and `data/splits/` are consumed directly by Person 3 (training).
