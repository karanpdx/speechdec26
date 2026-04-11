# Multimodal Neural Speech Decoding Pipeline

## What This Is

A two-stage multimodal neural speech decoding system for BCI research. Stage 1 encodes EEG, MEG, and/or fMRI signals recorded during speech perception into a learned shared embedding space and retrieves the most likely word via cosine similarity against BERT-embedded vocabulary. Stage 2 inverts this: given a word, it projects through the shared space and decodes back into predicted neural signals conditioned on subject identity.

## Core Value

A learned shared embedding space that genuinely aligns neural signals across modalities toward BERT word embeddings — verified by above-chance retrieval and positive cross-modal alignment gap.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Dataset inspection protocol followed for every new dataset before any preprocessing
- [ ] EEG preprocessing pipeline: raw → clean epoched tensors `(n_epochs, n_channels, n_timepoints)` with word labels
- [ ] MEG preprocessing pipeline: raw → clean epoched tensors with Maxwell/SSS filtering
- [ ] fMRI preprocessing pipeline: BOLD → per-word beta maps `(n_words, n_voxels)` via GLM
- [ ] Dataset alignment and reproducible train/val/test splits stratified by subject
- [ ] BERT vocabulary embeddings generated and saved for full vocabulary
- [ ] Modality encoders: EEGEncoder (EEGNet-style), MEGEncoder (shared temporal weights option), fMRIEncoder (MLP)
- [ ] SharedEmbeddingProjector and SubjectEmbedding modules
- [ ] Contrastive loss (CLIP-style), CrossModalAlignmentLoss, SubjectAdversarialLoss
- [ ] Stage 1 training loop: joint multi-modal contrastive training with adversarial subject de-identification
- [ ] Retrieval evaluation: top-1/5/10, MRR, cross-subject generalization, abstention curve
- [ ] Stage 1 evaluation report
- [ ] Stage 2 decoder architectures: EEGDecoder, MEGDecoder (shared base), fMRIDecoder
- [ ] Stage 2 training loop: frozen Stage 1 encoders, MSE + frequency-domain + spatial-smoothness losses
- [ ] Stage 2 evaluation: ERP component recovery, fMRI spatial correlation, round-trip consistency
- [ ] Stage 2 evaluation report
- [ ] Full integration checklist passes

### Out of Scope

- Fine-tuning BERT — BERT is frozen supervision throughout; its space is bridged via SharedEmbeddingProjector
- Live EEG domain shift correction — known failure mode, documented but not mitigated in v1
- Out-of-vocabulary word generalization — model is not trained on OOV words; failure is expected and documented
- Mobile or real-time inference — offline batch pipeline only
- Cross-dataset subject transfer — within-dataset subject holdout only for v1

## Context

- **Domain**: Cognitive neuroscience / BCI / representation learning
- **Paradigm**: Contrastive learning aligning neural modalities to BERT word embeddings; not borrowing BERT's space directly but learning an aligned shared space
- **Key architectural insight**: The shared embedding space is the core. Cross-modal alignment between modalities sharing stimulus labels (EEG↔MEG) is an explicit secondary objective
- **Subject variability**: Addressed via SubjectEmbedding (for Stage 2 conditioning) and SubjectAdversarialLoss (to make the shared space modality-informative but subject-agnostic)
- **Dataset uncertainty**: Exact datasets and formats are not predetermined. Every dataset goes through the inspection protocol before any preprocessing code is written
- **Evaluation philosophy**: Don't claim reconstruction "works" from MSE alone — validate ERP structure (N400) and fMRI spatial patterns explicitly

## Constraints

- **Stack**: Python 3.10+, PyTorch 2.x, MNE-Python, Nilearn, HuggingFace Transformers (BERT), scikit-learn — no deviations
- **Data integrity**: `data/raw/` is read-only; all transformations write to `data/processed/`
- **Agent discipline**: State before acting; spec before implementing; check ARCHITECTURE.md before every change; test every module
- **Shape logging**: Log tensor shapes at every preprocessing and model step (Python logging at INFO, not print)
- **No silent failures**: Surface blockers immediately; raise errors on format mismatches, don't attempt silent workarounds

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Learned shared space (not borrowed BERT space) | Direct BERT space use would ignore modality-specific structure; contrastive alignment lets the model discover what's shared | — Pending |
| CLIP-style symmetric contrastive loss | Learnable temperature, symmetric direction ensures both neural→text and text→neural alignment | — Pending |
| Subject adversarial loss | Prevents shared space from encoding subject identity, improving cross-subject generalization | — Pending |
| EEGNet architecture for EEG/MEG encoders | Well-validated for EEG; depthwise conv captures spatial+temporal structure compactly | — Pending |
| Stage 2 trains with frozen Stage 1 encoders | Prevents catastrophic forgetting of learned shared space | — Pending |
| Frequency-domain loss for EEG/MEG reconstruction | MSE alone doesn't enforce realistic oscillatory structure; PSD loss adds band-specific pressure | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-10 after initialization*
