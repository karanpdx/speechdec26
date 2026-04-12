# Spec: Dataset-Specific Preprocessing for `data-eeg` and `fmri`

## Scope

Update the EEG and fMRI preprocessing code so it correctly handles the concrete local datasets:

- `data-eeg/EEG Audio Dataset Preprocessing/*.mat`
- `fmri/ds002345-download/` (Narratives BIDS dataset)

The code must support these formats explicitly and fail clearly when the available files do not
contain the word-level labels required by the decoding pipeline.

## Inputs

- EEG `.mat` derivatives with:
  - one `data` struct per subject
  - `eeg`: 60 trial matrices, each `(3200, 66)`
  - `fsample.eeg = 64`
  - 66 channel labels, last two `EXG1`, `EXG2`
  - `event.eeg[*].value` binary codes `1`/`2`
- fMRI BIDS dataset `ds002345` with:
  - subject `sub-*` directories
  - `func/*_bold.nii.gz` and sidecar JSON with `RepetitionTime=1.5`
  - `*_events.tsv` files containing only story/music blocks, not word timings

## Outputs

- EEG preprocessing supports MATLAB derivative loading:
  - converts trials to `(n_trials, n_channels, n_timepoints)`
  - drops non-EEG channels like `EXG1`, `EXG2` by default
  - records correct `sfreq` and channel names
  - requires an explicit trial-to-word mapping before saving pipeline outputs
- fMRI preprocessing supports Narratives discovery:
  - loads BIDS runs and detects story-level-only events
  - searches optional external word-timing files if configured
  - raises a descriptive error when only sentence/story-level annotations are available

## Edge Cases

- EEG dataset may be partially downloaded, but the existing `.mat` derivatives are internally consistent.
- EEG event values `1`/`2` are not sufficient for a 10-word vocabulary; code must not silently use them as word labels.
- Narratives dataset may be large and partially downloaded by subject. Preprocessing should operate on the available subset.
- Narratives TSVs with labels like `story` and `music` must be treated as non-word annotations.
- If external TextGrid/alignment files are later added, the same fMRI preprocessor should accept them.

## Success Criteria

- Preprocessors recognize these dataset styles without format errors.
- EEG preprocessor can parse the local `.mat` derivatives and report their actual dimensions.
- fMRI preprocessor refuses to produce fake word labels from story-level TSVs.
- Dataset cards document that the currently available local data is insufficient for the 10-word shared vocabulary without additional alignment/label files.
