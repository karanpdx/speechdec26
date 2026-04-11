---
phase: 02-dataset-loader
plan: "02"
subsystem: worker_training/dataset
tags: [dataset, pytorch, collate, multimodal]
dependency_graph:
  requires: [MultiModalDataset (02-01)]
  provides: [collate_fn]
  affects: [build_shared_label_mask (02-03), training loop (phase 03)]
tech_stack:
  added: []
  patterns: [modality-grouped collate, graceful stub fallback]
key_files:
  created: []
  modified:
    - workers/worker_training/src/training/dataset.py
decisions:
  - "collate_fn catches NotImplementedError from build_shared_label_mask and falls back to all-False mask so the function is usable before plan 02-03 implements that stub"
metrics:
  duration: "~5 minutes"
  completed: "2026-04-10"
  tasks: 1
  files_modified: 1
---

# Phase 02 Plan 02: Implement collate_fn Summary

**One-liner:** Custom DataLoader collate function grouping multi-modal batch items by modality and stacking tensors, with graceful fallback for the build_shared_label_mask stub.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement collate_fn | 6df1f41 | src/training/dataset.py |

## What Was Built

`collate_fn` in `src/training/dataset.py`:

- Groups batch items by modality key into a `dict[str, list[dict]]`.
- For each modality present in the batch, stacks: `data` (Tensor), `label_idx` (LongTensor), `bert_emb` (Tensor), `subject_idx` (LongTensor), and collects `labels` as `list[str]`.
- Calls `build_shared_label_mask(batch)` and includes result as `shared_label_mask` key; catches `NotImplementedError` (stub for plan 02-03) and falls back to an all-False BoolTensor of length `len(batch)`.
- Modalities with no samples in the batch are absent from the returned dict.

Manual smoke test passed: batch of 3 items (2 EEG + 1 MEG) produced correct grouped output with proper tensor shapes.

## Deviations from Plan

**[Rule 2 - Missing critical functionality] Graceful fallback for build_shared_label_mask stub**
- **Found during:** Task 1
- **Issue:** build_shared_label_mask raises NotImplementedError (plan 02-03 stub); calling it directly would make collate_fn unusable until 02-03 completes.
- **Fix:** Wrapped call in try/except NotImplementedError, returning all-False mask as fallback.
- **Files modified:** workers/worker_training/src/training/dataset.py
- **Commit:** 6df1f41

## Known Stubs

| Stub | File | Reason |
|------|------|--------|
| `build_shared_label_mask` | src/training/dataset.py | Implemented in plan 02-03 |

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- SUMMARY.md created at workers/worker_training/.planning/phases/02-dataset-loader/02-02-SUMMARY.md
- Commit 6df1f41 exists (feat(02-02): implement collate_fn grouping by modality)
- collate_fn no longer raises NotImplementedError
