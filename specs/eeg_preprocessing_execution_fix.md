# EEG Preprocessing Execution Fix

## Scope
Fix ZuCo EEG preprocessing so one subject can run end-to-end and write a `.npz` file under `data/processed/zuco/eeg/`.

## Inputs
- ZuCo BrainVision header path `*_eeg.vhdr`
- Matching ZuCo events file `*_events.tsv`
- Existing `run_subject()` and `extract_epochs()` flow in `workers/worker_data/src/data/preprocess_eeg.py`

## Outputs
- EEG preprocessing resolves the correct events TSV for each BrainVision run
- One subject can complete preprocessing and save `sub-<id>_epochs.npz`

## Edge Cases
- `mne.io.read_raw_brainvision()` may populate `raw.filenames` with the `.eeg` data file instead of the `.vhdr` header path
- Existing callers that do not pass a source path should keep current behavior as a fallback

## Success Criterion
- Running `run_subject('sub-01', 'zuco', config)` completes without the observed parser error on the `.eeg` file and writes `data/processed/zuco/eeg/sub-01_epochs.npz`
