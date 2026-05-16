# BENJI — Multimodal Neural Speech Decoder

Built at the NeuroTech X Global Hackathon (Top Finalist). BENJI decodes intended speech 
from brain signals by fusing EEG, MEG, and fMRI into a shared semantic embedding space.

## What it does

Given multi-subject neural recordings during speech perception, BENJI maps each modality 
through a dedicated encoder into a shared 512-dim GloVe-anchored embedding space using 
supervised contrastive loss. At inference, the model predicts the most likely intended word 
from neural activity alone. We achieved 48% top-1 decoding accuracy on MEG (2x chance).

## Architecture

- **EEG/MEG:** EEGNet encoders per modality
- **fMRI:** ROI-based MLP encoder
- **Fusion:** Shared embedding space with supervised contrastive loss aligned to GloVe vectors
- **Output:** Ranked candidate words decoded from neural activity

## Datasets

- MASC-MEG (27 subjects, word/phoneme-level annotations)
- OpenNeuro ds006033 (simultaneous EEG + fMRI, inner speech)
- ZuCo 2.0 (18 subjects, 739 Wikipedia sentences)
- Huth fMRI (8 subjects, 27 narrative stories)

## My Role

I owned the dataset workstream: sourcing, downloading, and preprocessing all four datasets 
into a unified input format with temporal alignment across modalities with different sampling 
rates and spatial res
