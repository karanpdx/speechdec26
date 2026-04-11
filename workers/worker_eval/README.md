# Worker Eval — Person 4

## Your Job

Implement all evaluation metrics and report generators for both Stage 1 (retrieval) and Stage 2 (reconstruction). You produce the numbers that determine whether the pipeline actually works.

**You have almost no dependencies on other workers.** Your inputs are numpy arrays of embeddings — you never touch PyTorch models or raw data files. Use the synthetic data generators in `stubs/` to develop and test everything independently.

---

## What You Build

```
worker_eval/
├── src/evaluation/
│   ├── retrieval.py      # Stage 1 metrics: top-k accuracy, MRR, alignment gap, abstention
│   └── reconstruction.py # Stage 2 metrics: ERP, fMRI spatial corr, round-trip similarity
├── stubs/
│   └── generate_synthetic.py  # Synthetic embeddings + signals for testing
└── tests/
    ├── test_retrieval.py
    └── test_reconstruction.py
```

---

## Interface Contract — What You Receive

### From Person 3 (training checkpoints → exported numpy arrays)

After Stage 1 training, Person 3 will export:

```python
# Stage 1 retrieval inputs
neural_embeddings: np.ndarray   # (n_test_samples, embed_dim) — encoder output
text_embeddings:   np.ndarray   # (vocab_size, embed_dim)    — projector output
labels:            list[str]    # (n_test_samples,) ground truth words
vocab:             list[str]    # (vocab_size,) vocabulary
subject_ids:       list[str]    # (n_test_samples,) for per-subject metrics
```

After Stage 2 training, Person 3 will export:

```python
# Stage 2 reconstruction inputs
pred_eeg:    np.ndarray  # (n_test, n_channels, n_timepoints) — decoder output
real_eeg:    np.ndarray  # (n_test, n_channels, n_timepoints) — real data
pred_fmri:   np.ndarray  # (n_test_words, n_voxels)
real_fmri:   np.ndarray  # (n_test_words, n_voxels)
voxel_coords: np.ndarray # (n_voxels, 3) MNI coordinates
ch_names:    list[str]   # EEG channel names
sfreq:       float       # sampling frequency
```

**During development:** use `stubs/generate_synthetic.py` to generate all of these.

---

## Stage 1 Metrics

### `compute_retrieval_metrics(neural_embeddings, text_embeddings, labels, vocab, k_values)`
For each sample: cosine similarity against all vocab embeddings → rank → check top-k.
Returns: `{'top1': float, 'top5': float, 'top10': float, 'mrr': float}`

### `compute_cross_subject_generalization(neural_embeddings, text_embeddings, labels, vocab, subject_ids)`
Runs retrieval per subject. Returns per-subject metrics AND mean across subjects.
**Report mean across subjects, NOT across pooled samples.** Pooling inflates performance
for subjects with more data.

### `compute_cross_modal_alignment(eeg_embeddings, meg_embeddings, labels)`
For matched word pairs (same word, same subject if available):
- Cosine similarity of matched pairs
- Cosine similarity of random pairs (negative control)
- Report the gap: matched - random

A positive gap confirms the shared space is genuinely aligning modalities.

### `compute_abstention_curve(neural_embeddings, text_embeddings, labels, vocab, confidence_thresholds)`
For each threshold: abstain when (top1_score - top2_score) < threshold.
Returns coverage/accuracy tradeoff. Find threshold achieving 80% accuracy at max coverage.

---

## Stage 2 Metrics

### ERP Component Recovery (EEG/MEG)
1. Grand average ERP across all test words (real and predicted separately)
2. Plot real vs. predicted at centro-parietal channels (Pz, CPz, or equivalent)
3. N400 amplitude: mean in 300-500ms window minus -100 to 0ms baseline
4. Report Pearson r between real and predicted N400 amplitudes across words

### fMRI Spatial Correlation
1. Pearson r between predicted and real beta maps per word
2. Report mean ± std across words and subjects
3. Correlate predicted map with Neurosynth language localizer map

### Round-Trip Consistency
BERT embedding → SharedEmbeddingProjector → decoder → encoder → shared embedding
Compute cosine similarity between round-trip embedding and original.
Report mean ± std. Target: > 0.5 (> 0.7 indicates strong internal consistency).

---

## Report Outputs

**`evaluation/stage1_report.md`** containing:
- Per-modality top-1, top-5, top-10, MRR on test split
- Cross-subject generalization: per-subject top-1, mean, std
- Cross-modal alignment: matched vs. random cosine similarity
- Abstention curve: coverage/accuracy tradeoff table
- 10 representative failure cases (wrong prediction, top-3 shown)

**`evaluation/stage2_report.md`** containing:
- N400 correlation: Pearson r
- fMRI spatial correlation: mean ± std
- Neurosynth language map correlation
- Round-trip cosine similarity: mean ± std
- 5 good reconstructions, 5 failure cases

---

## Integration Handoff

When done, integration copies:
- `worker_eval/src/evaluation/` → `src/evaluation/`
- `worker_eval/tests/` → `tests/`
