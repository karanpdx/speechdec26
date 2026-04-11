# Preprocessing Execution Fixes Spec

## Scope
Make the existing preprocessing code runnable from the repository root without rewriting completed worker implementations.

## Inputs
- Existing worker implementations under `workers/worker_data/src/data/`
- Existing tests under `workers/worker_data/tests/test_preprocessing.py`
- Available datasets under `data/eeg_zuco`, `ag3kj/osfstorage`, and `data/fmri_huth`

## Outputs
- Importable root package `src.data.*`
- Completed alignment and vocabulary embedding utilities
- fMRI preprocessing that can consume Huth word timings from available `TextGrid` files when TSVs are absent
- Real generated outputs under `data/processed/...` and `data/splits/...` when upstream data and dependencies are sufficient

## Edge Cases
- `ARCHITECTURE.md` is absent at repo root: do not invent architectural changes; keep implementation compatible with current worker layout.
- Hugging Face model weights may need to be loaded from local cache only for tests and output generation.
- Some modalities may not produce enough samples or may fail on dataset-specific artifacts; failures must be explicit and non-faked.
- Huth datasets may lack TSV event files but include `TextGrid` word tiers; support this fallback only when the word tier is present and parseable.

## Success Criteria
- `./venv/bin/python -m pytest workers/worker_data/tests/test_preprocessing.py -q` passes.
- EEG and MEG preprocessing run against real local data and produce `.npz` outputs if the current code/data combination supports them.
- fMRI preprocessing either produces outputs using available timings or fails clearly with the exact missing requirement.
- Split JSON and vocab embeddings are generated if processed modality outputs are available.
