"""
Stage 1 training loop — joint multi-modal contrastive training.

Trains all modality encoders and the SharedEmbeddingProjector together
using contrastive loss, cross-modal alignment, and subject adversarial loss.

Entry point: scripts/train_stage1.py
"""

import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_models(config: dict) -> dict:
    """
    Instantiate all Stage 1 models based on config.

    Attempts to import from src.models.encoders; falls back to
    stubs.model_stubs with a warning if src.models is not available.

    Args:
        config: Parsed train_stage1.yaml config dict.

    Returns:
        Dict with keys: 'eeg_encoder', 'meg_encoder', 'fmri_encoder',
        'projector', 'subject_emb', 'adversarial_loss'.
    """
    try:
        from src.models.encoders import EEGEncoder, MEGEncoder, fMRIEncoder
        from src.models.projector import SharedEmbeddingProjector
        from src.models.subject import SubjectEmbedding
        logger.info("build_models: using src.models encoders")
    except ImportError:
        logger.warning("build_models: src.models not available — using stubs/model_stubs.py")
        from stubs.model_stubs import (
            EEGEncoder, MEGEncoder, fMRIEncoder,
            SharedEmbeddingProjector, SubjectEmbedding,
        )

    n_subjects = config.get("n_subjects") or 20

    eeg_encoder = EEGEncoder(
        n_channels=config["eeg_channels"],
        n_timepoints=config["eeg_timepoints"],
        embed_dim=config["embed_dim"],
    )
    meg_encoder = MEGEncoder(
        n_channels=config["meg_channels"],
        n_timepoints=config["meg_timepoints"],
        embed_dim=config["embed_dim"],
    )
    fmri_encoder = fMRIEncoder(
        n_voxels=config["fmri_voxels"],
        embed_dim=config["embed_dim"],
    )
    projector = SharedEmbeddingProjector(
        bert_dim=768,
        embed_dim=config["embed_dim"],
    )
    subject_emb = SubjectEmbedding(
        n_subjects=n_subjects,
        subject_embed_dim=config["subject_embed_dim"],
    )

    from src.training.losses import SubjectAdversarialLoss
    adversarial_loss = SubjectAdversarialLoss(
        embed_dim=config["embed_dim"],
        n_subjects=n_subjects,
    )

    device = torch.device(config.get("device", "cpu"))
    models = {
        "eeg_encoder": eeg_encoder.to(device),
        "meg_encoder": meg_encoder.to(device),
        "fmri_encoder": fmri_encoder.to(device),
        "projector": projector.to(device),
        "subject_emb": subject_emb.to(device),
        "adversarial_loss": adversarial_loss.to(device),
    }
    return models


def build_optimizer(models: dict, config: dict) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer over all trainable parameters.

    Args:
        models: Output of build_models().
        config: Parsed train_stage1.yaml.

    Returns:
        torch.optim.AdamW optimizer.
    """
    params = []
    for key in ("eeg_encoder", "meg_encoder", "fmri_encoder", "projector", "subject_emb"):
        params += list(models[key].parameters())
    # adversarial_loss has its own internal classifier — include it too
    params += list(models["adversarial_loss"].parameters())
    return torch.optim.AdamW(
        params,
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


def compute_alpha(current_epoch: int, total_epochs: int) -> float:
    """
    Compute gradient reversal alpha for current epoch.

    Linearly increases from 0 to 1 over total_epochs.

    Args:
        current_epoch: Current epoch index (0-based).
        total_epochs: Total training epochs.

    Returns:
        float in [0, 1].
    """
    if total_epochs <= 1:
        return 1.0
    return current_epoch / (total_epochs - 1)


def train_one_epoch(
    models: dict,
    dataloader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    csv_writer,
) -> dict:
    """
    Run one full training epoch.

    For each batch:
        1. Forward each present modality encoder
        2. Forward SharedEmbeddingProjector for BERT embeddings
        3. ContrastiveLoss per modality — sum
        4. CrossModalAlignmentLoss for modality pairs with shared labels
        5. SubjectAdversarialLoss (alpha linearly scheduled)
        6. Total loss = sum + lambda_cross * cross_modal + lambda_adv * adversarial
        7. Backward, gradient clip (max_norm=1.0), optimizer step

    Logs to CSV every log_every_n_steps steps.

    Args:
        models: Output of build_models().
        dataloader: Training DataLoader.
        optimizer: AdamW optimizer.
        config: Parsed config dict.
        epoch: Current epoch (0-based).
        csv_writer: csv.DictWriter for logging.

    Returns:
        Dict of mean epoch losses: total, eeg, meg, fmri, cross_modal, adversarial.
    """
    raise NotImplementedError


def validate(models: dict, val_dataloader, vocab_embeddings, vocab: list[str]) -> dict:
    """
    Run retrieval evaluation on the validation split.

    Calls compute_retrieval_metrics from src.evaluation.retrieval
    (or a stub if not yet available).

    Args:
        models: Output of build_models().
        val_dataloader: Validation DataLoader.
        vocab_embeddings: (V, embed_dim) numpy array.
        vocab: List of vocabulary word strings.

    Returns:
        Dict: {'top1_eeg', 'top5_eeg', 'top1_meg', 'top5_meg', 'top1_fmri', 'top5_fmri'}
        Only keys for present modalities are included.
    """
    raise NotImplementedError


def save_checkpoint(
    models: dict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
    is_best: bool = False,
) -> str:
    """
    Save model checkpoint.

    Always saves as epoch_{N}.pt.
    If is_best=True, also saves as best.pt (overwriting previous best).

    Args:
        models: Output of build_models().
        optimizer: Optimizer state.
        epoch: Current epoch.
        metrics: Val metrics dict.
        checkpoint_dir: Directory to save checkpoints.
        is_best: Whether this is the best val checkpoint so far.

    Returns:
        Path to saved checkpoint.
    """
    raise NotImplementedError


def train(config_path: str) -> None:
    """
    Full Stage 1 training run.

    Loads config, builds models and dataloaders, trains for n_epochs,
    validates every val_every_n_epochs, saves checkpoints.

    Does NOT train Stage 2 or touch decoders.

    Args:
        config_path: Path to configs/train_stage1.yaml.
    """
    raise NotImplementedError
