# Spec: S3 Data Staging Workflow

## Scope

Allow the repo to work with raw datasets stored in S3 by staging them into local directories
before preprocessing and training. The codebase should remain local-filesystem based.

## Inputs

- S3 bucket and optional prefix
- S3 subfolders:
  - `eeg-data`
  - `fmri-data`
  - `masc_meg`
- Local sync target root such as `data/raw/`

## Outputs

- A config file mapping each S3 dataset folder to a local staging directory
- A CLI that runs `aws s3 sync` for selected modalities
- Preprocessing configs that point at the staged local directories by default

## Constraints

- Do not rewrite preprocessing or training to stream directly from S3.
- Keep the existing data loaders using normal `Path`, `glob`, `np.load`, MNE, and nibabel APIs.
- Support either full `s3://...` sources or bucket/prefix-relative folder names.

## Edge Cases

- The S3 folder may contain an extra nested dataset directory inside the modality folder.
- The local dataset tree may differ slightly from the defaults; configs must stay editable.
- AWS CLI may not be installed or configured; fail clearly and tell the user what is missing.

## Success Criteria

- `python scripts/sync_s3_data.py --config configs/s3_data.yaml --dry-run` prints valid sync commands.
- After syncing, preprocessing configs can point at the staged local dataset roots without code changes.
