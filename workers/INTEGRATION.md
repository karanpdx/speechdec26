# Integration Guide

How to assemble the four workers into the full pipeline once everyone is done.

---

## Prerequisites

Before integrating, confirm the following from each worker:

**Person 1 (worker_data):**
- [ ] All tests in `worker_data/tests/test_preprocessing.py` pass
- [ ] At least one real dataset processed: `data/processed/<dataset>/` exists with `.npz` files
- [ ] `data/splits/split_v1.json` exists and `verify_split_integrity()` passes
- [ ] `data/processed/vocab_embeddings.npz` exists and is non-empty

**Person 2 (worker_models):**
- [ ] All tests in `worker_models/tests/test_encoders.py` pass
- [ ] All tests in `worker_models/tests/test_decoders.py` pass
- [ ] Temporal weight sharing test passes

**Person 3 (worker_training):**
- [ ] All tests in `worker_training/tests/test_losses.py` pass
- [ ] Training loop runs for 5 epochs without NaN loss (using stub data)

**Person 4 (worker_eval):**
- [ ] All tests in `worker_eval/tests/test_retrieval.py` pass
- [ ] All tests in `worker_eval/tests/test_reconstruction.py` pass

---

## Step 1: Copy source modules into canonical src/ structure

```bash
# From repo root

# Person 1 — data pipeline
cp -r workers/worker_data/src/data/          src/data/
cp -r workers/worker_data/configs/           configs/

# Person 2 — model architectures
cp -r workers/worker_models/src/models/      src/models/

# Person 3 — training
cp -r workers/worker_training/src/training/  src/training/
cp -r workers/worker_training/scripts/       scripts/
cp -r workers/worker_training/configs/train_stage1.yaml  configs/
cp -r workers/worker_training/configs/train_stage2.yaml  configs/

# Person 4 — evaluation
cp -r workers/worker_eval/src/evaluation/    src/evaluation/
```

---

## Step 2: Update imports in training modules

Person 3's training modules were developed with stub imports. After copying, update:

```python
# In src/training/train_stage1.py and train_stage2.py
# Change:
from stubs.model_stubs import EEGEncoder, ...
# To:
from src.models.encoders import EEGEncoder, ...
from src.models.decoders import EEGDecoder, ...
```

The dataset loading code should already point to the real data paths from config.

---

## Step 3: Copy tests

```bash
cp workers/worker_data/tests/test_preprocessing.py    tests/
cp workers/worker_models/tests/test_encoders.py       tests/
cp workers/worker_models/tests/test_decoders.py       tests/
cp workers/worker_training/tests/test_losses.py       tests/
cp workers/worker_eval/tests/test_retrieval.py        tests/
cp workers/worker_eval/tests/test_reconstruction.py   tests/
```

---

## Step 4: Run the full test suite

```bash
pytest tests/ -v
```

All tests must pass before proceeding to pipeline execution.

---

## Step 5: Run Stage 1 training

```bash
# Update configs/train_stage1.yaml with your dataset's values:
#   n_subjects, eeg_channels, meg_channels, fmri_voxels, etc.

python scripts/train_stage1.py --config configs/train_stage1.yaml
```

Monitor `checkpoints/stage1/training_log.csv` for loss curves.
Training is not stuck at NaN = good sign.

---

## Step 6: Validate Stage 1

```bash
# Export test-set embeddings from the best checkpoint
# (add an export script in scripts/export_embeddings.py)

# Run Stage 1 evaluation
python -c "
from src.evaluation.retrieval import compute_retrieval_metrics, compute_cross_modal_alignment
import numpy as np
# Load exported embeddings...
"
```

Check:
- [ ] Val top-1 accuracy is above chance (> 1/vocab_size) for at least one modality
- [ ] Cross-modal alignment gap > 0 (matched > random)

Do NOT proceed to Stage 2 until Stage 1 passes validation.

---

## Step 7: Run Stage 2 training

```bash
# Ensure stage1_checkpoint in configs/train_stage2.yaml points to best.pt

python scripts/train_stage2.py --config configs/train_stage2.yaml
```

Verify at start of training:
- Log message confirming encoder weights are frozen
- No AssertionError from verify_encoders_frozen()

---

## Step 8: Run the full integration checklist

From AGENTS.md:

**Data pipeline:**
- [ ] Dataset cards exist at `data/processed/<name>/DATASET_CARD.md`
- [ ] All preprocessed `.npz` files pass output validation assertions
- [ ] Split file verified (no subject leakage)
- [ ] Vocabulary embeddings saved and non-empty

**Stage 1:**
- [ ] All encoder tests pass
- [ ] All loss function tests pass
- [ ] Training runs without NaN loss for at least 5 epochs
- [ ] Validation retrieval accuracy above chance for at least one modality
- [ ] Cross-modal alignment gap positive
- [ ] Stage 1 evaluation report generated

**Stage 2:**
- [ ] All decoder tests pass
- [ ] Encoder weights confirmed frozen during Stage 2 training
- [ ] Reconstruction MSE decreases over training
- [ ] ERP comparison figure generated and visually plausible
- [ ] Round-trip cosine similarity > 0.5
- [ ] Stage 2 evaluation report generated

---

## Troubleshooting

### Import errors after copying
Ensure `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`,
`src/training/__init__.py`, and `src/evaluation/__init__.py` all exist (can be empty).

```bash
touch src/__init__.py src/data/__init__.py src/models/__init__.py \
      src/training/__init__.py src/evaluation/__init__.py
```

### Shape mismatch errors
The most common integration bug. Check:
1. `eeg_channels` in `train_stage1.yaml` matches `n_channels` in preprocessing config
2. `fmri_voxels` matches `n_pca_components` (if PCA was applied) or total voxel count
3. `n_timepoints` = `int((epoch_tmax - epoch_tmin) * target_sfreq) + 1`

### NaN loss at start of Stage 1
Usually temperature instability in ContrastiveLoss. Check:
- Temperature parameter is properly clamped
- Input embeddings are L2-normalized inside the loss
- Batch size is > 1

### Stage 2 MSE not decreasing
- Confirm encoders are actually frozen (check the log)
- Reduce learning rate in `train_stage2.yaml`
- Check decoder output shapes match real signal shapes
