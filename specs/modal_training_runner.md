# Spec: Modal Training Runner

## Scope

Add a Modal app that can run the existing root pipeline entry points remotely without creating a second training path.

Supported operations:

- alignment
- stage1
- stage2

The Modal app should call the same underlying code used by local execution:

- `src.data.align_splits.run_alignment`
- `src.training.train_stage1.train`
- `src.training.train_stage2.train`

## Inputs

- Existing root configs under `configs/`
- Existing root wrappers under `src/`
- Existing local training assumptions:
  - relative `data/...` paths
  - relative `checkpoints/...` paths

## Outputs

- `scripts/modal_train.py`
- Modal app with:
  - code mounted into `/workspace`
  - `data/` and `checkpoints/` mounted as Modal volumes
  - local entrypoint dispatching `alignment`, `stage1`, or `stage2`

## Constraints

- Do not duplicate training logic.
- Keep Stage 1 and Stage 2 separate on Modal just as they are locally.
- The module must still import cleanly when `modal` is not installed, so repo tests can run offline.

## Edge Cases

- `modal` is absent locally: import should not crash; calling the local entrypoint should raise a clear error.
- Stage 2 requires `stage1_checkpoint`.
- Relative config paths must resolve correctly inside the Modal container.

## Success Criteria

- The new script imports cleanly in local tests.
- It exposes a single local entrypoint for `modal run scripts/modal_train.py ...`.
- Alignment, Stage 1, and Stage 2 all reuse the existing root pipeline code.
