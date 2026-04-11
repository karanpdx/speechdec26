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
    raise NotImplementedError


def build_optimizer(models: dict, config: dict) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer over all trainable parameters.

    Args:
        models: Output of build_models().
        config: Parsed train_stage1.yaml.

    Returns:
        torch.optim.AdamW optimizer.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
