# Spec: Split Path Contract for Training

## Scope

Make the split file produced by `worker_data` resolvable by the training loader in
`worker_training` without assuming a single hard-coded processed-data directory layout.

## Inputs

- Processed modality directories configured in `configs/splits.yaml`
- Split entries written to `data/splits/split_v1.json`
- `MultiModalDataset(processed_base_dir=...)`

## Outputs

- Split JSON stores stable, non-absolute paths for each processed `.npz` file.
- Split JSON also records the common processed-data base directory used to resolve those paths.
- `MultiModalDataset` supports both:
  - new relative-path split entries
  - legacy filename-only split entries

## Edge Cases

- If processed modality directories only share `data/processed`, the split entries may include
  dataset subdirectories like `dataset_a/eeg/sub-01_epochs.npz`.
- If processed data for a modality already sits directly under `<processed_base_dir>/<modality>/`,
  legacy filename-only entries must still load.
- Absolute paths should be accepted defensively, but newly generated splits should not emit them.

## Success Criteria

- `run_alignment()` writes a split JSON that training can consume without manual path rewriting.
- `MultiModalDataset` resolves files correctly for both old and new split formats.
- Worker-local tests and root import/smoke checks continue to pass.
