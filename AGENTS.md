# AGENTS.md — Multimodal Neural Speech Decoding Pipeline

This document is the authoritative instruction set for all agents working on this project.
Read it fully before taking any action. Do not skip sections. Do not assume context
that is not stated here.

---

## Project Overview

We are building a two-stage multimodal neural speech decoding pipeline.

**Stage 1 (Decoding):** Neural signals from EEG, MEG, and/or fMRI recorded while a subject
processes speech are encoded by modality-specific encoders into a shared embedding space,
then decoded to retrieve the most likely word or phrase via cosine retrieval against BERT
embeddings of a candidate vocabulary.

**Stage 2 (Reconstruction):** A word or phrase is projected into the shared embedding space
and then decoded back into predicted neural signals in each modality, conditioned on subject
identity. The reconstruction should recover physiologically meaningful structure (e.g., N400
timing in EEG/MEG, language ROI activation in fMRI).

The shared embedding space is the architectural core. It must be learned — not borrowed
directly from BERT — via contrastive training that aligns all modalities toward BERT embeddings
of stimulus words, plus explicit cross-modal alignment between modalities that share stimulus
labels.

**Stack:** Python 3.10+, PyTorch 2.x, MNE-Python (EEG/MEG preprocessing), Nilearn (fMRI
preprocessing), HuggingFace Transformers (BERT), scikit-learn (baseline models, PCA).

**Repository structure expected:**

```
/
├── AGENTS.md               # this file
├── ARCHITECTURE.md         # system architecture document (auto-updated)
├── data/
│   ├── raw/                # raw downloaded datasets, never modified
│   ├── processed/          # preprocessed outputs, one subfolder per modality per dataset
│   └── splits/             # train/val/test splits as JSON index files
├── src/
│   ├── data/               # dataset inspection, loading, and preprocessing modules
│   ├── models/             # encoder, decoder, shared space, subject embedding modules
│   ├── training/           # training loops, loss functions, optimizers
│   ├── evaluation/         # retrieval, reconstruction metrics, visualization
│   └── utils/              # logging, checkpointing, config
├── configs/                # YAML config files per experiment
├── scripts/                # CLI entry points (train, eval, preprocess)
├── tests/                  # unit and integration tests
├── notebooks/              # exploratory analysis only, never production code
└── specs/                  # per-component spec files
```

---

## Agent Discipline Rules (Non-Negotiable)

These apply to every agent at every step regardless of task scope.

1. **State before acting.** Before any implementation step, state in one sentence what you
   are about to do and why. Do not take silent actions.

2. **Spec before implementing.** For any non-trivial function, class, or module, write a
   brief spec (inputs, outputs, edge cases, success criterion) before writing code. Save
   specs to `specs/` as markdown files.

3. **Check ARCHITECTURE.md before every change.** If an implementation decision is not
   consistent with the architecture document, surface the conflict and resolve it explicitly
   before proceeding. If the architecture needs to change, update `ARCHITECTURE.md` and note
   what changed and why.

4. **Test every module.** Every module in `src/` has a corresponding test in `tests/`.
   Tests cover the happy path, at least one edge case, and at least one expected failure.
   Tests must pass before a module is considered done.

5. **Never modify raw data.** Files in `data/raw/` are read-only. All transformations
   write to `data/processed/` with a clear naming convention.

6. **Log shapes at every stage.** Neural data shapes are a common source of silent bugs.
   At every preprocessing and model step, log the shape of the tensor being processed.
   Use Python `logging` (not print) at INFO level.

7. **Surface blockers immediately.** If a step cannot be completed because of a missing
   dependency, unclear spec, or data issue, stop and report the blocker with enough detail
   to resolve it. Do not attempt workarounds without surfacing them.

8. **Dataset format is unknown until inspected.** The team will supply datasets but their
   exact format is not predetermined. Every dataset agent task begins with inspection before
   any preprocessing code is written. See the Dataset Inspection Protocol below.

---

## Dataset Inspection Protocol

Because multiple datasets may be used per modality and their formats are not known in advance,
every new dataset must be inspected before any preprocessing code is written.

**Run this inspection for every dataset before writing any loading or preprocessing code:**

### Step 1: Directory and File Audit

```python
import os
import pathlib

def audit_dataset(root_path: str):
    root = pathlib.Path(root_path)
    # Print directory tree up to 3 levels deep
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth <= 3:
            indent = "  " * (depth - 1)
            print(f"{indent}{p.name}  [{p.stat().st_size // 1024} KB]" if p.is_file() else f"{indent}{p.name}/")
    
    # Count files by extension
    from collections import Counter
    ext_counts = Counter(p.suffix for p in root.rglob("*") if p.is_file())
    print("\nFile types:", dict(ext_counts))
```

Report: directory structure, file types present, approximate total size.

### Step 2: Format Identification

Identify the data format. Common formats for neural datasets:

- `.fif` — MNE raw/epochs file (EEG or MEG). Load with `mne.io.read_raw_fif()` or
  `mne.read_epochs()`.
- `.edf` / `.bdf` — EDF/BDF format (EEG). Load with `mne.io.read_raw_edf()`.
- `.set` / `.fdt` — EEGLAB format. Load with `mne.io.read_raw_eeglab()`.
- `.mat` — MATLAB file. Load with `scipy.io.loadmat()` or `h5py` for v7.3+.
- `.npy` / `.npz` — NumPy arrays. Load with `np.load()`.
- `.nii` / `.nii.gz` — NIfTI fMRI volumes. Load with `nibabel.load()`.
- BIDS format — directory structure with `sub-XX/ses-XX/eeg/` or `func/` subdirectories
  and accompanying `*_events.tsv` files for stimulus timing.
- `.hdf5` / `.h5` — HDF5. Load with `h5py`.

For each dataset, document: format, loading library, whether it is BIDS-compliant.

### Step 3: Signal Inspection

For EEG/MEG datasets:

```python
import mne

raw = mne.io.read_raw_fif("path/to/file.fif", preload=False)
print("Sampling frequency:", raw.info['sfreq'])
print("Number of channels:", len(raw.ch_names))
print("Channel names (first 10):", raw.ch_names[:10])
print("Channel types:", set(raw.get_channel_types()))
print("Duration (s):", raw.times[-1])
print("Has events:", mne.find_events(raw) if 'stim' in raw.get_channel_types() else "check annotations")
```

For fMRI datasets:

```python
import nibabel as nib
import numpy as np

img = nib.load("path/to/file.nii.gz")
print("Shape:", img.shape)          # (x, y, z, n_timepoints)
print("Voxel size:", img.header.get_zooms())
print("TR:", img.header.get_zooms()[3] if len(img.shape) == 4 else "N/A")
print("Affine:", img.affine)
```

### Step 4: Stimulus/Label Inspection

This is the most critical step. The pipeline requires word-level stimulus labels with
precise timing. Inspect how labels are stored:

- BIDS: look for `*_events.tsv` files. Expected columns: `onset`, `duration`, `trial_type`.
  `trial_type` should contain the word or stimulus identifier.
- MNE annotations: `raw.annotations` — check `description` field.
- MATLAB: inspect all fields of the struct. Look for fields named `events`, `triggers`,
  `labels`, `stimuli`, `words`.
- Separate CSV/TSV: check for accompanying metadata files in the dataset root.

Report: where labels are stored, what the label format is (word string, numeric code,
or stimulus ID that requires a separate lookup table), timing precision (is onset in
samples or seconds), and whether the label maps to a single word or a sentence/phrase.

**If labels are sentence-level rather than word-level**, document this clearly. The
preprocessing pipeline will need to either (a) use sentence-level epochs with the full
sentence as the text target, or (b) apply forced alignment (using MFA or wav2vec) to
get word-level onsets from the audio. Do not silently assume one or the other.

### Step 5: Subject and Session Structure

Document:
- Number of subjects
- Number of sessions per subject
- Whether subjects and sessions are balanced
- Whether there is a separate test set or held-out subjects

### Step 6: Write the Dataset Card

After inspection, produce a dataset card at `data/processed/<dataset_name>/DATASET_CARD.md`:

```markdown
# Dataset Card: <dataset_name>

## Format
- File format: 
- BIDS compliant: yes/no
- Loading library: 

## Signal Properties
- Modality: EEG / MEG / fMRI
- Sampling rate: (EEG/MEG only)
- Number of channels/voxels: 
- TR: (fMRI only)

## Subjects and Sessions
- N subjects: 
- N sessions per subject: 
- Held-out test subjects: 

## Stimulus/Label Structure
- Label granularity: word / sentence / phrase
- Label format: string / numeric code (with lookup table at <path>)
- Timing precision: samples / seconds
- Onset stored in: <location>
- Notes on label quality: 

## Known Issues
- (list any quirks, missing data, bad channels, etc.)

## Loading Example
```python
# minimal working example to load one subject's data
```
```

Do not proceed to preprocessing until the dataset card is complete and reviewed.

---

## Stage 1: Decoding Pipeline

### Agent 1A: EEG Preprocessor

**Responsibility:** Transform raw EEG recordings into clean, epoched, normalized tensors
of shape `(n_epochs, n_channels, n_timepoints)` with corresponding word labels.

**Inputs:**
- Raw EEG dataset in `data/raw/<dataset_name>/`
- Dataset card confirming format and label structure
- Config: `configs/preprocessing_eeg.yaml`

**Config schema:**
```yaml
dataset_name: str
subjects: list[str] | "all"
l_freq: float          # high-pass cutoff (default 0.1 Hz)
h_freq: float          # low-pass cutoff (default 40.0 Hz)
notch_freqs: list[float]  # e.g. [50.0] or [60.0]
epoch_tmin: float      # epoch start relative to word onset (default -0.1 s)
epoch_tmax: float      # epoch end relative to word onset (default 0.6 s)
baseline: [float, float] | null  # baseline window (default [-0.1, 0.0])
target_sfreq: float    # resample to this rate (default 256 Hz)
n_ica_components: int  # ICA components (default 20)
reject_threshold: float | null  # peak-to-peak rejection threshold in V
output_dir: str
```

**Processing steps (in order):**

1. Load raw file using the appropriate MNE loader identified in the dataset card.
2. Select only EEG channels: `raw.pick_types(eeg=True)`.
3. Set montage if not already set. Use standard 10-20 if montage is unknown.
4. Apply bandpass filter: `raw.filter(l_freq, h_freq, method='firwin')`.
5. Apply notch filter at specified frequencies.
6. Resample to `target_sfreq` if current sfreq differs.
7. Fit ICA on a copy of the raw data. Use `mne.preprocessing.ICA` with
   `n_components=n_ica_components`, `method='fastica'`. Automatically identify and
   exclude eye blink and cardiac components using `ica.find_bads_eog()` and
   `ica.find_bads_ecg()` where reference channels exist. Log which components were
   excluded. Apply ICA to raw.
8. Extract events from annotations or stimulus channel depending on dataset format.
   Map event codes to word labels using the lookup table from the dataset card.
9. Epoch around word onsets: `mne.Epochs(raw, events, tmin=epoch_tmin, tmax=epoch_tmax,
   baseline=baseline, reject=reject_threshold, preload=True)`.
10. Drop bad epochs. Log the number of epochs dropped and why.
11. Z-score normalize each channel across all epochs for this subject:
    `(epochs - mean) / std` where mean and std are computed across the time and epoch
    dimensions per channel.
12. Save output as `data/processed/<dataset_name>/eeg/sub-<id>_epochs.npz` containing:
    - `data`: float32 array `(n_epochs, n_channels, n_timepoints)`
    - `labels`: list of word strings, length `n_epochs`
    - `subject_id`: str
    - `sfreq`: float
    - `ch_names`: list of str
    - `event_onsets_s`: float array of onset times in seconds

**Edge cases to handle:**
- Missing ICA reference channels: skip EOG/ECG detection, log a warning, proceed without
  artifact removal. Do not crash.
- Epoch count below 10 for a subject: log a warning and skip that subject. Do not include
  in output.
- Label is sentence-level rather than word-level: raise a clear error asking for
  clarification on whether to use sentence embeddings or apply forced alignment.
- Dataset has multiple sessions: concatenate epochs across sessions per subject, preserving
  session ID as metadata.

**Output validation:**
```python
assert data.dtype == np.float32
assert data.shape == (n_epochs, n_channels, n_timepoints)
assert len(labels) == n_epochs
assert not np.any(np.isnan(data))
assert not np.any(np.isinf(data))
# Check z-score normalization is approximately correct
assert abs(data.mean()) < 0.1
assert abs(data.std() - 1.0) < 0.2
```

---

### Agent 1B: MEG Preprocessor

**Responsibility:** Same as Agent 1A but for MEG data, accounting for MEG-specific
preprocessing steps.

**Differences from EEG preprocessing:**

- Channel selection: pick both magnetometers and gradiometers:
  `raw.pick_types(meg=True)`. If the dataset has both, keep both and note the channel
  count separately.
- Apply Maxwell filtering (SSS) before ICA if the dataset was recorded on an Elekta/MEGIN
  system: `mne.preprocessing.maxwell_filter(raw)`. This requires a `crosstalk_file` and
  `fine_cal_file` if available in the dataset. If not available, apply `tsss` mode.
  Log whether SSS was applied.
- Gradiometer and magnetometer channels have different units and scales. After picking,
  apply `mne.preprocessing.maxwell_filter` or manually scale: divide magnetometers by
  their typical scale (~1e-12 T) and gradiometers by theirs (~1e-11 T/m) before
  z-scoring. The z-score normalization handles this automatically if applied per-channel,
  but log the pre-normalization scale.
- ICA component rejection: MEG is more sensitive to head movement artifacts. Check for
  movement artifacts using `mne.preprocessing.annotate_movement()` if head position data
  is available.

**Output format:** identical to EEG preprocessor but `ch_names` will reflect MEG channel
names. Save to `data/processed/<dataset_name>/meg/sub-<id>_epochs.npz`.

---

### Agent 1C: fMRI Preprocessor

**Responsibility:** Transform raw fMRI BOLD timeseries into per-word beta maps of shape
`(n_words, n_voxels)` using a GLM with HRF convolution.

**Inputs:**
- Raw fMRI NIfTI files in `data/raw/<dataset_name>/`
- Events TSV or equivalent label file (word onsets in seconds)
- Config: `configs/preprocessing_fmri.yaml`

**Config schema:**
```yaml
dataset_name: str
subjects: list[str] | "all"
tr: float                    # repetition time in seconds
hrf_model: str               # 'spm', 'glover', or 'fir' (default 'spm')
smoothing_fwhm: float        # spatial smoothing kernel in mm (default 6.0)
high_pass: float             # high-pass filter cutoff in Hz (default 0.01)
mask: str | null             # path to brain mask NIfTI, or null to auto-compute
roi_mask: str | null         # optional: restrict to language ROI mask
n_pca_components: int | null # if set, apply PCA after masking
output_dir: str
```

**Processing steps (in order):**

1. Load the BOLD NIfTI: `nibabel.load()`. Confirm shape is `(x, y, z, n_timepoints)`.
   Log the shape and TR.
2. Load brain mask. If `mask` is null, compute a mask using `nilearn.masking.compute_brain_mask()`.
3. Load events. Construct a pandas DataFrame with columns `onset`, `duration`, `trial_type`
   where `trial_type` is the word string. If events are in a different format, convert
   explicitly and document the conversion.
4. Apply spatial smoothing: `nilearn.image.smooth_img(img, fwhm=smoothing_fwhm)`.
5. Fit a first-level GLM using `nilearn.glm.first_level.FirstLevelModel`:
   ```python
   from nilearn.glm.first_level import FirstLevelModel
   glm = FirstLevelModel(
       t_r=tr,
       hrf_model=hrf_model,
       high_pass=high_pass,
       mask_img=mask,
       smoothing_fwhm=None,  # already smoothed
   )
   glm.fit(bold_img, events_df)
   ```
6. Extract a beta map per unique word: for each unique word in `trial_type`, compute the
   contrast map (beta coefficients) using `glm.compute_contrast(word, output_type='effect_size')`.
   This gives a NIfTI image per word.
7. Apply the brain mask to each beta map to get a 1D vector: `nilearn.masking.apply_mask()`.
8. If `roi_mask` is specified, apply it to further restrict voxels.
9. If `n_pca_components` is set, fit PCA on the (n_words x n_voxels) matrix and project
   down. Save the PCA object for later reconstruction. Note: PCA must be fit on training
   subjects only and applied to validation/test subjects using the already-fit transform.
10. Save output as `data/processed/<dataset_name>/fmri/sub-<id>_betas.npz` containing:
    - `data`: float32 array `(n_words, n_voxels_or_components)`
    - `labels`: list of word strings, length `n_words`
    - `subject_id`: str
    - `voxel_coords`: int array `(n_voxels, 3)` of MNI coordinates (for visualization)
    - `pca_explained_variance`: float (if PCA was applied)

**Edge cases:**
- Multiple runs per subject: concatenate events across runs, fit one GLM per run, average
  beta maps across runs for the same word.
- Word appears only once per subject: the beta estimate will be noisy. Log a warning if
  any word appears fewer than 3 times across all runs for a subject.
- Missing volumes or motion outliers: add confound regressors from `fmriprep` outputs if
  available (motion parameters, FD scrubbing). If not available, log a warning and proceed
  without confound regression.

**Output validation:**
```python
assert data.dtype == np.float32
assert data.shape == (n_words, n_voxels)
assert len(labels) == n_words
assert not np.any(np.isnan(data))
# Beta maps should not be all zeros
assert data.std() > 0
```

---

### Agent 1D: Dataset Alignment and Split Generator

**Responsibility:** Align processed outputs across modalities and datasets, ensure consistent
vocabulary, and generate reproducible train/val/test splits stratified by subject.

**Inputs:**
- All processed `.npz` files from Agents 1A, 1B, 1C
- Config: `configs/splits.yaml`

**Config schema:**
```yaml
val_subjects: list[str]   # subject IDs held out for validation
test_subjects: list[str]  # subject IDs held out for test (never seen during training)
min_word_freq: int        # minimum occurrences of a word across training set to include (default 5)
vocab_source: str         # 'intersection' (words present in all modalities) or 'union'
seed: int                 # random seed for any shuffling
```

**Processing steps:**

1. Load all processed files. Build a vocabulary: the set of unique word labels across all
   processed files.
2. Apply `min_word_freq` filter: remove words that appear fewer than `min_word_freq` times
   in the training subjects across all modalities combined.
3. If `vocab_source` is `intersection`, restrict to words present in at least one sample
   from each modality. If `union`, keep all words that meet the frequency threshold.
4. Generate BERT embeddings for the full vocabulary:
   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch

   tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   model = AutoModel.from_pretrained('bert-base-uncased')
   model.eval()

   def get_bert_embedding(word: str) -> np.ndarray:
       inputs = tokenizer(word, return_tensors='pt')
       with torch.no_grad():
           outputs = model(**inputs)
       # Use [CLS] token embedding
       return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

   vocab_embeddings = {word: get_bert_embedding(word) for word in vocabulary}
   ```
   Save to `data/processed/vocab_embeddings.npz`.

5. Generate splits as JSON index files:
   ```json
   {
     "train": {"eeg": ["sub-01_epochs.npz", ...], "meg": [...], "fmri": [...]},
     "val": {"eeg": [...], "meg": [...], "fmri": [...]},
     "test": {"eeg": [...], "meg": [...], "fmri": [...]}
   }
   ```
   Save to `data/splits/split_v1.json`.

6. Print split statistics: number of subjects, number of epochs/words, vocabulary size,
   per-modality counts. Log any modality with zero samples in a split.

**Invariants to check:**
- No test subject appears in train or val
- No val subject appears in train
- Vocabulary is identical across all splits (labels outside vocabulary are excluded, not
  remapped)

---

### Agent 1E: Model Implementation — Encoders and Shared Space

**Responsibility:** Implement all encoder architectures and the shared embedding space
module. No training logic here — pure model definitions.

**File:** `src/models/encoders.py`

**Implement the following classes:**

#### `EEGEncoder`
```python
class EEGEncoder(nn.Module):
    """
    EEGNet-style encoder for EEG epochs.
    Input: (batch, n_channels, n_timepoints)
    Output: (batch, embed_dim)
    """
```
Architecture: depthwise conv2d across channels (spatial filter), followed by depthwise
conv1d across time (temporal filter), average pooling, separable conv1d, average pooling,
flatten, linear projection to `embed_dim`, LayerNorm.

Parameters: `n_channels`, `n_timepoints`, `embed_dim=768`, `F1=8` (temporal filters),
`D=2` (depth multiplier), `F2=16` (pointwise filters), `dropout=0.25`.

Reference architecture: Lawhern et al. 2018 (EEGNet). Implement from scratch, do not
import from external EEGNet packages (they may not be maintained).

#### `MEGEncoder`
Identical architecture to `EEGEncoder` but accepts different `n_channels`. The temporal
convolution weights may optionally be shared with `EEGEncoder` if `share_temporal=True`
is passed to both constructors. In that case, both encoders must reference the same
`nn.Parameter` object for the temporal filter weights. Implement this sharing explicitly.

#### `fMRIEncoder`
```python
class fMRIEncoder(nn.Module):
    """
    MLP encoder for fMRI beta maps.
    Input: (batch, n_voxels)
    Output: (batch, embed_dim)
    """
```
Architecture: Linear(n_voxels, 2048) + ReLU + Dropout, Linear(2048, 1024) + ReLU +
Dropout, Linear(1024, embed_dim) + LayerNorm.

Parameters: `n_voxels`, `embed_dim=768`, `dropout=0.3`.

If `n_voxels > 10000`, add a warning log recommending PCA preprocessing.

#### `SharedEmbeddingProjector`
```python
class SharedEmbeddingProjector(nn.Module):
    """
    Small MLP that projects BERT embeddings into the learned shared space.
    Used to produce target embeddings during contrastive training, and as
    the entry point for Stage 2.
    Input: (batch, bert_dim=768)
    Output: (batch, embed_dim)
    """
```
Architecture: Linear(bert_dim, embed_dim) + ReLU + Linear(embed_dim, embed_dim) + LayerNorm.

This is trained jointly with the encoders. Its purpose: BERT's space and the learned shared
space are related but not identical. This projector bridges them.

#### `SubjectEmbedding`
```python
class SubjectEmbedding(nn.Module):
    """
    Lookup table of per-subject embedding vectors.
    Used to condition Stage 2 decoders.
    """
```
Wraps `nn.Embedding(n_subjects, subject_embed_dim=64)`. Exposes a method
`get_mean_embedding()` that returns the mean of all subject embeddings — used as a
generic prior for unseen subjects.

**Implementation requirements:**
- All modules use `nn.Module` properly (no functional-only implementations).
- Forward methods include shape assertions in debug mode: wrap with
  `if torch.is_grad_enabled(): assert ...` so assertions do not fire during inference.
- Every class has a docstring stating input shape, output shape, and any assumptions.
- No global state. All configuration is passed to `__init__`.

**Tests (`tests/test_encoders.py`):**
```python
def test_eeg_encoder_output_shape():
    enc = EEGEncoder(n_channels=64, n_timepoints=175, embed_dim=768)
    x = torch.randn(8, 64, 175)
    out = enc(x)
    assert out.shape == (8, 768)

def test_meg_encoder_output_shape():
    enc = MEGEncoder(n_channels=306, n_timepoints=175, embed_dim=768)
    x = torch.randn(8, 306, 175)
    out = enc(x)
    assert out.shape == (8, 768)

def test_fmri_encoder_output_shape():
    enc = fMRIEncoder(n_voxels=1000, embed_dim=768)
    x = torch.randn(8, 1000)
    out = enc(x)
    assert out.shape == (8, 768)

def test_temporal_weight_sharing():
    eeg_enc = EEGEncoder(n_channels=64, n_timepoints=175, share_temporal=True)
    meg_enc = MEGEncoder(n_channels=306, n_timepoints=175, share_temporal=True)
    eeg_enc.share_temporal_weights(meg_enc)
    # Modifying one should affect the other
    assert eeg_enc.temporal_conv.weight.data_ptr() == meg_enc.temporal_conv.weight.data_ptr()
```

---

### Agent 1F: Loss Functions

**Responsibility:** Implement all training loss functions as standalone, testable classes.

**File:** `src/training/losses.py`

#### `ContrastiveLoss`
CLIP-style symmetric contrastive loss between neural embeddings and text embeddings.

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature_init: float = 0.07, learnable_temp: bool = True):
        ...
    
    def forward(self, neural_emb: Tensor, text_emb: Tensor) -> Tensor:
        """
        neural_emb: (batch, embed_dim) — L2 normalized
        text_emb: (batch, embed_dim) — L2 normalized
        Returns scalar loss.
        """
```

Implementation: compute cosine similarity matrix `(batch x batch)` by matrix multiply of
L2-normalized embeddings. Scale by `exp(temperature)`. Apply cross-entropy in both
directions (neural→text and text→neural). Return mean.

The temperature is a learnable `nn.Parameter` if `learnable_temp=True`. Clamp it to
`[log(0.01), log(100)]` to prevent instability.

Both `neural_emb` and `text_emb` must be L2-normalized before the similarity matrix is
computed. Normalize inside the loss function, do not assume normalized inputs.

#### `CrossModalAlignmentLoss`
Contrastive loss between two different modality embeddings that share stimulus labels.

```python
class CrossModalAlignmentLoss(nn.Module):
    def forward(
        self,
        emb_a: Tensor,          # (batch, embed_dim)
        emb_b: Tensor,          # (batch, embed_dim)
        shared_label_mask: Tensor  # (batch,) bool: True if this sample has a match in both modalities
    ) -> Tensor:
```

Only compute the loss over samples where `shared_label_mask` is True. If fewer than 2
such samples exist in the batch, return `torch.tensor(0.0, requires_grad=True)` and log
a warning. Do not crash.

#### `SubjectAdversarialLoss`
Gradient reversal loss that makes the shared embedding uninformative about subject identity.

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class SubjectAdversarialLoss(nn.Module):
    def __init__(self, embed_dim: int, n_subjects: int, hidden_dim: int = 128):
        # small classifier: embed_dim -> hidden_dim -> n_subjects
        ...
    
    def forward(self, shared_emb: Tensor, subject_ids: Tensor, alpha: float = 1.0) -> Tensor:
        """
        shared_emb: (batch, embed_dim)
        subject_ids: (batch,) int
        alpha: gradient reversal scale (increase over training)
        Returns cross-entropy loss of subject classifier (reversed in backward).
        """
```

**Tests (`tests/test_losses.py`):**
```python
def test_contrastive_loss_diagonal_is_minimum():
    # With perfect embeddings (identity pairs), loss should be lower than with shuffled pairs
    emb = F.normalize(torch.randn(16, 768), dim=-1)
    loss_matched = ContrastiveLoss()(emb, emb)
    shuffled = emb[torch.randperm(16)]
    loss_shuffled = ContrastiveLoss()(emb, shuffled)
    assert loss_matched < loss_shuffled

def test_contrastive_loss_with_batch_size_1_raises():
    # Contrastive loss is undefined for batch size 1
    emb = F.normalize(torch.randn(1, 768), dim=-1)
    with pytest.raises(ValueError):
        ContrastiveLoss()(emb, emb)

def test_cross_modal_loss_empty_mask_returns_zero():
    emb_a = torch.randn(8, 768)
    emb_b = torch.randn(8, 768)
    mask = torch.zeros(8, dtype=torch.bool)
    loss = CrossModalAlignmentLoss()(emb_a, emb_b, mask)
    assert loss.item() == 0.0
```

---

### Agent 1G: Training Loop — Stage 1

**Responsibility:** Implement the Stage 1 training loop that trains all encoders jointly
using the combined loss.

**File:** `src/training/train_stage1.py`

**Entry point:** `scripts/train_stage1.py` which loads config and calls the training module.

**Config schema (`configs/train_stage1.yaml`):**
```yaml
# Data
split_file: data/splits/split_v1.json
vocab_embeddings: data/processed/vocab_embeddings.npz
modalities: [eeg, meg, fmri]  # which modalities to include

# Model
embed_dim: 768
eeg_channels: 64
eeg_timepoints: 175
meg_channels: 306
meg_timepoints: 175
fmri_voxels: 1000
subject_embed_dim: 64
share_temporal_weights: false

# Loss weights
lambda_cross_modal: 0.5
lambda_subject_adversarial: 0.1
adversarial_alpha_schedule: linear  # increase alpha from 0 to 1 over training

# Training
batch_size: 64
lr: 3e-4
weight_decay: 1e-4
n_epochs: 50
warmup_epochs: 5
grad_clip: 1.0
device: cuda

# Logging
checkpoint_dir: checkpoints/stage1/
log_every_n_steps: 10
val_every_n_epochs: 5
```

**Training loop requirements:**

1. Build a `MultiModalDataset` that samples batches containing samples from all available
   modalities. Each batch should contain roughly equal numbers of samples per modality.
   Do not require all modalities to be present for every word -- handle missingness by
   computing only the losses for modalities present in a given batch.

2. For each batch:
   a. Forward pass through each modality's encoder for samples of that modality.
   b. Forward pass through `SharedEmbeddingProjector` for the corresponding BERT embeddings.
   c. Compute `ContrastiveLoss` per modality. Sum them.
   d. Compute `CrossModalAlignmentLoss` for any pairs of modalities with shared stimulus
      labels in the batch (check by matching `word_label` strings). Use a `shared_label_mask`
      indicating which samples have a counterpart in another modality.
   e. Compute `SubjectAdversarialLoss` on all shared embeddings with their subject IDs.
      Scale by `alpha` which increases linearly from 0 to 1 over training.
   f. Total loss = sum of modality contrastive losses + lambda_cross * cross_modal_loss
      + lambda_adv * adversarial_loss.
   g. Backward pass, gradient clipping, optimizer step.

3. Validation (every `val_every_n_epochs`): run retrieval evaluation (Agent 1H) on the
   val split. Log top-1 and top-5 accuracy per modality.

4. Save checkpoints: best val top-1 accuracy, and every 10 epochs regardless.

5. Log to a CSV: epoch, step, total_loss, loss_eeg, loss_meg, loss_fmri, loss_cross_modal,
   loss_adversarial, val_top1_eeg, val_top1_meg, val_top1_fmri.

**Do not implement the dataloader here.** The dataloader is Agent 1D's responsibility --
call the loading utilities from `src/data/`. If they do not exist yet, raise an ImportError
with a clear message.

---

### Agent 1H: Evaluation — Retrieval and Generalization

**Responsibility:** Implement all Stage 1 evaluation metrics and produce the required
evaluation report.

**File:** `src/evaluation/retrieval.py`

**Metrics to implement:**

#### `compute_retrieval_metrics`
```python
def compute_retrieval_metrics(
    neural_embeddings: np.ndarray,   # (n_samples, embed_dim)
    text_embeddings: np.ndarray,     # (vocab_size, embed_dim)
    labels: list[str],               # (n_samples,) ground truth word
    vocab: list[str],                # (vocab_size,) vocabulary
    k_values: list[int] = [1, 5, 10]
) -> dict:
    """
    Returns dict with keys: top1, top5, top10, mrr (mean reciprocal rank).
    """
```

Implementation: for each sample, compute cosine similarity against all vocab embeddings,
rank, check if ground truth word is in top-k. MRR = mean of 1/rank across all samples.

#### `compute_cross_subject_generalization`
Runs `compute_retrieval_metrics` separately for each held-out test subject. Returns
per-subject metrics and mean across subjects. The mean across subjects (not across
samples pooled) is the reported generalization number, because pooling inflates performance
for subjects with more samples.

#### `compute_cross_modal_alignment`
For samples where the same word was processed in two different modalities (matched by
word label and subject if available):
- Compute cosine similarity between EEG embedding and MEG embedding for matched pairs.
- Compare to cosine similarity between randomly paired embeddings (negative control).
- Report mean similarity for matched pairs, mean for random pairs, and the gap.

A large gap (matched >> random) indicates the shared space is genuinely aligning modalities,
not just coincidentally projecting to nearby regions due to the shared text target.

#### `compute_abstention_curve`
```python
def compute_abstention_curve(
    neural_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list[str],
    vocab: list[str],
    confidence_thresholds: np.ndarray  # e.g. np.linspace(0, 1, 50)
) -> dict:
    """
    For each threshold, abstain when (top1_score - top2_score) < threshold.
    Returns: coverage (fraction not abstained), accuracy on non-abstained samples.
    """
```

This produces the abstention curve: as threshold increases, coverage decreases and
accuracy increases. Report the threshold that achieves 80% accuracy with maximum coverage.

**Evaluation report:** produce `evaluation/stage1_report.md` containing:
- Per-modality top-1, top-5, top-10, MRR on the test split
- Cross-subject generalization: per-subject top-1, mean, std
- Cross-modal alignment: matched vs. random cosine similarity
- Abstention curve: coverage/accuracy tradeoff table
- Representative failure cases: 10 examples where the model predicted the wrong word,
  with the top-3 predictions shown

---

## Stage 2: Reconstruction Pipeline

### Agent 2A: Model Implementation — Decoders

**Responsibility:** Implement the modality-specific decoder architectures for Stage 2.

**File:** `src/models/decoders.py`

#### `EEGDecoder`
```python
class EEGDecoder(nn.Module):
    """
    Reconstructs an EEG epoch from a shared embedding + subject embedding.
    Input: shared_emb (batch, embed_dim), subject_emb (batch, subject_embed_dim)
    Output: (batch, n_channels, n_timepoints)
    """
```

Architecture:
1. Concatenate `shared_emb` and `subject_emb` → (batch, embed_dim + subject_embed_dim)
2. Linear(embed_dim + subject_embed_dim, 1024) + ReLU
3. Linear(1024, 2048) + ReLU
4. Reshape to (batch, n_channels, 2048 // n_channels) -- a compressed time representation
5. ConvTranspose1d to upsample time dimension progressively to `n_timepoints`
6. BatchNorm + ReLU after each transposed conv
7. Final Conv1d(channels, n_channels, 1) to mix channel information
8. Output shape: (batch, n_channels, n_timepoints)

The number of transposed conv layers and their strides should be computed automatically
from `n_timepoints` in `__init__` to avoid hardcoding.

#### `MEGDecoder`
Identical architecture to `EEGDecoder` with different `n_channels`. Do not copy-paste --
`MEGDecoder` should inherit from a common `_TemporalDecoder` base class that both share.

#### `fMRIDecoder`
```python
class fMRIDecoder(nn.Module):
    """
    Reconstructs an fMRI beta map from a shared embedding + subject embedding.
    Input: shared_emb (batch, embed_dim), subject_emb (batch, subject_embed_dim)
    Output: (batch, n_voxels)
    """
```

Architecture: MLP only (no transposed convolutions -- no time dimension to reconstruct).
Linear(embed_dim + subject_embed_dim, 1024) + ReLU + Dropout(0.3),
Linear(1024, 2048) + ReLU + Dropout(0.3),
Linear(2048, n_voxels).

No activation on the output -- beta maps can be positive or negative.

**Tests (`tests/test_decoders.py`):**
```python
def test_eeg_decoder_output_shape():
    dec = EEGDecoder(embed_dim=768, subject_embed_dim=64, n_channels=64, n_timepoints=175)
    shared = torch.randn(4, 768)
    subj = torch.randn(4, 64)
    out = dec(shared, subj)
    assert out.shape == (4, 64, 175)

def test_fmri_decoder_output_shape():
    dec = fMRIDecoder(embed_dim=768, subject_embed_dim=64, n_voxels=1000)
    shared = torch.randn(4, 768)
    subj = torch.randn(4, 64)
    out = dec(shared, subj)
    assert out.shape == (4, 1000)
```

---

### Agent 2B: Training Loop — Stage 2

**Responsibility:** Train the Stage 2 decoders with the Stage 1 encoders frozen.

**File:** `src/training/train_stage2.py`

**Critical:** Load Stage 1 checkpoint and freeze all encoder weights before training.
Verify they are frozen:
```python
for param in eeg_encoder.parameters():
    assert not param.requires_grad, "Encoder weights must be frozen for Stage 2 training"
```

**Config schema (`configs/train_stage2.yaml`):**
```yaml
stage1_checkpoint: checkpoints/stage1/best.pt
split_file: data/splits/split_v1.json
vocab_embeddings: data/processed/vocab_embeddings.npz

# Loss weights
mse_weight: 1.0
freq_domain_weight: 0.1      # frequency-domain loss for EEG/MEG
spatial_smooth_weight: 0.05  # spatial smoothness for fMRI

batch_size: 32
lr: 1e-4
n_epochs: 30
device: cuda
checkpoint_dir: checkpoints/stage2/
```

**Loss functions for Stage 2:**

1. **MSE loss** (primary): between predicted and real signal, per modality.

2. **Frequency domain loss** (EEG/MEG only): compute FFT of predicted and real waveforms
   along the time dimension. Penalize difference in power spectral density across bands:
   delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz). Weight bands
   equally. This encourages realistic oscillatory structure.
   ```python
   def freq_domain_loss(pred: Tensor, target: Tensor, sfreq: float) -> Tensor:
       pred_fft = torch.fft.rfft(pred, dim=-1)
       target_fft = torch.fft.rfft(target, dim=-1)
       # PSD is magnitude squared
       pred_psd = pred_fft.abs() ** 2
       target_psd = target_fft.abs() ** 2
       return F.mse_loss(pred_psd, target_psd)
   ```

3. **Spatial smoothness loss** (fMRI only): penalize the L2 norm of differences between
   adjacent voxels in the predicted beta map. Requires voxel coordinate information to
   define adjacency. Use the `voxel_coords` from the preprocessed output to build a
   sparse adjacency matrix at startup, then compute the smoothness penalty per batch.

---

### Agent 2C: Evaluation — Reconstruction Quality

**Responsibility:** Evaluate Stage 2 reconstruction quality and produce the validation
visualizations.

**File:** `src/evaluation/reconstruction.py`

**Metrics and visualizations to implement:**

#### ERP Component Recovery (EEG/MEG)
For each test subject and each word in the test set:
1. Get predicted EEG epoch from the decoder.
2. Get real EEG epoch from the test data.
3. Compute the grand average ERP across all words (real and predicted separately).
4. Plot real vs. predicted ERP waveforms at centro-parietal channels (Pz, CPz, or
   equivalent in the dataset's montage). Save as `evaluation/figures/erp_comparison.png`.
5. Compute the N400 amplitude: mean amplitude in the 300-500ms window relative to the
   -100 to 0ms baseline. Report correlation between real and predicted N400 amplitudes
   across words.

#### Spatial Activation Recovery (fMRI)
1. For each word, get predicted and real beta maps.
2. Compute Pearson correlation between predicted and real beta maps across voxels.
   Report mean and std across words and subjects.
3. Correlate the predicted beta map with a meta-analytic language localizer map.
   Obtain the language localizer from Neurosynth programmatically:
   ```python
   # Use nilearn's fetch_atlas_destrieux or a pre-downloaded Neurosynth map
   # for the term 'language'
   ```
   Report the correlation as evidence that the decoder recovers language-relevant spatial
   patterns.
4. Produce a glass brain visualization of mean predicted activation using
   `nilearn.plotting.plot_glass_brain`. Save as `evaluation/figures/fmri_glass_brain.png`.

#### Round-Trip Consistency
For each test word:
1. Get the BERT embedding of the word.
2. Project through `SharedEmbeddingProjector` to get a shared embedding.
3. Decode through each modality decoder to get predicted signals.
4. Encode predicted signals back through the corresponding encoder.
5. Compute cosine similarity between the round-trip embedding and the original shared
   embedding.
6. Report mean round-trip cosine similarity across words. A value above 0.7 indicates
   the pipeline is internally consistent.

#### Reconstruction Report
Produce `evaluation/stage2_report.md` containing:
- N400 correlation: Pearson r between real and predicted N400 amplitudes
- fMRI spatial correlation: mean ± std across words
- Neurosynth language map correlation
- Round-trip cosine similarity: mean ± std
- Representative examples: 5 words where reconstruction is good, 5 where it fails

---

## Integration Checklist

Before declaring the full pipeline complete, verify all of the following:

**Data pipeline:**
- [ ] Dataset cards exist for all datasets at `data/processed/<name>/DATASET_CARD.md`
- [ ] All preprocessed `.npz` files pass their output validation assertions
- [ ] Split file exists and is verified (no subject leakage)
- [ ] Vocabulary embeddings saved and confirmed non-empty

**Stage 1:**
- [ ] All encoder tests pass
- [ ] All loss function tests pass
- [ ] Training runs without NaN loss for at least 5 epochs
- [ ] Validation retrieval accuracy is above chance (chance = 1/vocab_size) for at least
  one modality
- [ ] Cross-modal alignment gap (matched > random) is positive
- [ ] Stage 1 evaluation report generated

**Stage 2:**
- [ ] All decoder tests pass
- [ ] Encoder weights confirmed frozen during Stage 2 training
- [ ] Reconstruction MSE decreases over training (not stuck at initialization)
- [ ] ERP comparison figure generated and visually plausible (not flat lines)
- [ ] Round-trip cosine similarity > 0.5
- [ ] Stage 2 evaluation report generated

**Known acceptable failure modes (document, do not fix):**
- Live EEG domain shift from dataset to headset: expected. Mitigated by subject-level
  Euclidean alignment at inference.
- fMRI reconstruction spatial correlation < 0.3: expected for a first pass. Report it
  honestly.
- Round-trip similarity drops significantly for OOV (out-of-vocabulary) words: expected.
  The model was not trained on them.

---

## What Not To Do

- Do not train Stage 2 before Stage 1 has converged and been validated.
- Do not fine-tune BERT. It is frozen throughout. BERT is supervision, not a trainable component.
- Do not pool subjects for cross-subject generalization metrics. Report per-subject, then mean.
- Do not report top-1 accuracy on the training split as a performance metric.
- Do not silently handle a dataset format mismatch. Raise an error and write the dataset card.
- Do not claim the reconstruction "works" based only on MSE decrease. Validate the ERP
  and spatial pattern recovery explicitly.
