# Spec: Worker Training Stage 2 Training Loop

## Scope

Implement `workers/worker_training/src/training/train_stage2.py` so the worker training package can train Stage 2 decoders from frozen Stage 1 encoders.

## Inputs

- Stage 2 config YAML with:
  - `stage1_checkpoint`
  - `split_file`
  - `vocab_embeddings`
  - model dimensions for EEG, MEG, fMRI, and subject embeddings
  - optimizer, logging, and loss weight settings
- Stage 1 checkpoint produced by `train_stage1.save_checkpoint()`
- `MultiModalDataset` / `collate_fn` batches with modality-specific tensors, BERT embeddings, and subject indices
- Decoder classes from `src.models.decoders` or `stubs.model_stubs`

## Outputs

- Frozen Stage 1 encoders and projector loaded from checkpoint
- Trainable Stage 2 decoders and subject embedding
- Stage 2 checkpoint files written to `checkpoint_dir`
- CSV log written to `checkpoint_dir/train_log.csv`
- Per-epoch loss dicts containing:
  - `total`
  - `mse`
  - `freq`
  - `smooth`
  - `eeg`
  - `meg`
  - `fmri`

## Function Contracts

### `load_stage1_checkpoint(checkpoint_path, models, device)`

- Loads checkpoint weights into the provided Stage 1 models
- Freezes all encoder parameters and projector parameters
- Leaves subject embedding trainable for Stage 2
- Returns a dict with keys `eeg`, `meg`, `fmri`
- Raises `FileNotFoundError` if the checkpoint does not exist
- Raises `KeyError` if checkpoint contents are missing expected model state

### `verify_encoders_frozen(encoders)`

- Asserts every encoder parameter has `requires_grad=False`
- Raises `AssertionError` immediately if any encoder parameter is still trainable

### `freq_domain_loss(pred, target, sfreq)`

- Accepts `(batch, channels, time)` tensors
- Computes band-wise PSD MSE over delta, theta, alpha, beta bands using `torch.fft.rfft`
- Returns a scalar tensor
- Returns approximately zero for identical signals
- Preserves gradient flow to `pred`

### `spatial_smoothness_loss(pred, adjacency_indices)`

- Accepts `(batch, voxels)` predictions and `(edges, 2)` adjacency pairs
- Penalizes squared differences across adjacent voxels
- Returns a scalar tensor
- Returns differentiable zero when there are no adjacency edges

### `build_voxel_adjacency(voxel_coords, max_dist_mm)`

- Accepts `(n_voxels, 3)` coordinates
- Returns `(n_edges, 2)` long tensor with upper-triangular adjacent voxel pairs
- Includes pairs whose Euclidean distance is `<= max_dist_mm`
- Returns empty `(0, 2)` tensor when there are fewer than two voxels or no edges

### `build_decoders(config)`

- Instantiates EEG, MEG, and fMRI decoders from `src.models.decoders`
- Falls back to `stubs.model_stubs` with a warning if real models are unavailable
- Moves decoders to `config["device"]`

### `train_one_epoch(...)`

- Verifies encoders are frozen before processing
- Runs encoders under `torch.no_grad()`
- Uses encoder-produced shared embeddings and trainable subject embeddings to reconstruct per modality
- Applies:
  - MSE to all modalities
  - frequency-domain loss to EEG and MEG only
  - spatial smoothness loss to fMRI only
- Backpropagates only through decoders and subject embedding
- Returns mean losses for the epoch
- Logs tensor shapes at INFO level for each modality encountered

### `train(config_path)`

- Loads config
- Builds Stage 1 models, loads and freezes checkpoint, builds decoders
- Builds train/val dataloaders with `MultiModalDataset`
- Precomputes fMRI voxel adjacency once from the training set when fMRI is enabled
- Optimizes decoder + subject embedding parameters
- Writes CSV logs and Stage 2 checkpoints
- Runs validation epochs with the same loss computation but without optimizer steps

## Edge Cases

- Missing checkpoint path: fail loudly with `FileNotFoundError`
- Missing real model package: use stubs fallback instead of crashing
- No fMRI data present: skip adjacency build and smoothness loss
- Empty adjacency tensor: smoothness loss is zero, not an error
- Batch missing one or more modalities: compute losses only for present modalities
- Identical predicted and target temporal signals: frequency-domain loss is near zero

## Success Criteria

- `train_stage2.py` no longer contains `NotImplementedError`
- Stage 2 loss helper tests pass
- New Stage 2 training tests cover:
  - happy path
  - edge case
  - expected failure
- Existing Stage 1 worker_training tests still pass
