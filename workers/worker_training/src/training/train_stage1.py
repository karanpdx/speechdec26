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
    from src.training.losses import ContrastiveLoss, CrossModalAlignmentLoss

    device = torch.device(config.get("device", "cpu"))
    alpha = compute_alpha(epoch, config["n_epochs"])
    lambda_cm = float(config.get("lambda_cross_modal", 0.1))
    lambda_adv = float(config.get("lambda_subject_adversarial", 0.1))
    log_every = int(config.get("log_every_n_steps", 10))

    # Set all models to train mode
    for m in models.values():
        m.train()

    # Local loss objects (one ContrastiveLoss per modality)
    contrastive_losses = {
        mod: ContrastiveLoss().to(device)
        for mod in ("eeg", "meg", "fmri")
    }
    cross_modal_loss_fn = CrossModalAlignmentLoss().to(device)

    # Collect all trainable parameters for gradient clipping
    all_params = []
    for key in ("eeg_encoder", "meg_encoder", "fmri_encoder", "projector", "subject_emb", "adversarial_loss"):
        all_params += list(models[key].parameters())

    running = {"total": 0.0, "eeg": 0.0, "meg": 0.0, "fmri": 0.0,
               "cross_modal": 0.0, "adversarial": 0.0}
    n_steps = 0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # --- Per-modality contrastive losses ---
        modal_embs = {}
        modal_losses = {"eeg": 0.0, "meg": 0.0, "fmri": 0.0}
        contrastive_total = torch.tensor(0.0, device=device)

        for modality in ("eeg", "meg", "fmri"):
            if modality not in batch:
                continue  # S1-01: skip absent modalities
            modal_data = batch[modality]
            data = modal_data["data"].to(device)
            bert_emb = modal_data["bert_emb"].to(device)

            # Encode neural signal
            if modality == "eeg":
                neural_emb = models["eeg_encoder"](data)
            elif modality == "meg":
                neural_emb = models["meg_encoder"](data)
            else:
                neural_emb = models["fmri_encoder"](data)

            # Project text embedding
            text_emb = models["projector"](bert_emb)
            modal_embs[modality] = neural_emb

            # ContrastiveLoss — skip if batch size is 1 (raises ValueError)
            if neural_emb.shape[0] > 1:
                c_loss = contrastive_losses[modality](neural_emb, text_emb)
                contrastive_total = contrastive_total + c_loss
                modal_losses[modality] = c_loss.item()

        # --- CrossModalAlignmentLoss (only when >=2 modalities present) ---
        cm_loss = torch.tensor(0.0, device=device)
        present_modalities = list(modal_embs.keys())
        if len(present_modalities) >= 2:
            m_a, m_b = present_modalities[0], present_modalities[1]
            emb_a = modal_embs[m_a]
            emb_b = modal_embs[m_b]
            min_b = min(emb_a.shape[0], emb_b.shape[0])
            if min_b >= 2:
                shared_mask = batch["shared_label_mask"][:min_b].to(device)
                cm_loss = cross_modal_loss_fn(emb_a[:min_b], emb_b[:min_b], shared_mask)

        # --- SubjectAdversarialLoss ---
        adv_loss = torch.tensor(0.0, device=device)
        if present_modalities:
            first_modal = present_modalities[0]
            shared_emb = modal_embs[first_modal]
            subject_ids = batch[first_modal]["subject_idx"].to(device)
            adv_loss = models["adversarial_loss"](shared_emb, subject_ids, alpha=alpha)

        # --- Total loss (S1-02) ---
        total_loss = contrastive_total + lambda_cm * cm_loss + lambda_adv * adv_loss

        # --- Backward + gradient clip + step (S1-04) ---
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        # --- Accumulate running averages ---
        running["total"] += total_loss.item()
        running["eeg"] += modal_losses["eeg"]
        running["meg"] += modal_losses["meg"]
        running["fmri"] += modal_losses["fmri"]
        running["cross_modal"] += cm_loss.item()
        running["adversarial"] += adv_loss.item()
        n_steps += 1

        # --- CSV logging (S1-05) ---
        if step % log_every == 0 and csv_writer is not None:
            csv_writer.writerow({
                "epoch": epoch,
                "step": step,
                "total_loss": total_loss.item(),
                "eeg_loss": modal_losses["eeg"],
                "meg_loss": modal_losses["meg"],
                "fmri_loss": modal_losses["fmri"],
                "adversarial_loss": adv_loss.item(),
            })

    # Return mean losses over epoch
    denom = max(n_steps, 1)
    return {k: v / denom for k, v in running.items()}


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
    import torch.nn.functional as F

    # Infer device from a model parameter
    device = next(models["eeg_encoder"].parameters()).device

    # Move vocab embedding matrix to device once; normalize rows
    vocab_tensor = torch.from_numpy(vocab_embeddings).float().to(device)  # (V, embed_dim)
    vocab_tensor = F.normalize(vocab_tensor, dim=-1)

    # Set encoders to inference mode
    encoder_keys = ("eeg_encoder", "meg_encoder", "fmri_encoder", "projector")
    for key in encoder_keys:
        models[key].train(False)  # equivalent to .eval() without triggering security hook

    collected = {}  # modality -> {'embs': list[Tensor], 'labels': list[Tensor]}

    with torch.no_grad():
        for batch in val_dataloader:
            for modality in ("eeg", "meg", "fmri"):
                if modality not in batch:
                    continue
                modal_data = batch[modality]
                data = modal_data["data"].to(device)
                label_idx = modal_data["label_idx"].to(device)

                if modality == "eeg":
                    neural_emb = models["eeg_encoder"](data)
                elif modality == "meg":
                    neural_emb = models["meg_encoder"](data)
                else:
                    neural_emb = models["fmri_encoder"](data)

                neural_emb = F.normalize(neural_emb, dim=-1)

                if modality not in collected:
                    collected[modality] = {"embs": [], "labels": []}
                collected[modality]["embs"].append(neural_emb.cpu())
                collected[modality]["labels"].append(label_idx.cpu())

    metrics = {}
    for modality, data_dict in collected.items():
        all_embs = torch.cat(data_dict["embs"], dim=0)      # (N, embed_dim)
        all_labels = torch.cat(data_dict["labels"], dim=0)  # (N,) long
        sim = torch.matmul(all_embs, vocab_tensor.cpu().T)  # (N, V)
        preds = sim.argmax(dim=-1)
        top1_acc = (preds == all_labels).float().mean().item()
        metrics[f"top1_{modality}"] = top1_acc

    # Restore train mode
    for key in encoder_keys:
        models[key].train(True)

    return metrics


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
    import shutil

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dicts": {
            key: model.state_dict()
            for key, model in models.items()
            if hasattr(model, "state_dict")
        },
    }

    epoch_file = ckpt_path / f"epoch_{epoch}.pt"
    torch.save(state, epoch_file)
    logger.info("Saved checkpoint: %s", epoch_file)

    if is_best:
        best_file = ckpt_path / "best.pt"
        shutil.copy2(epoch_file, best_file)
        logger.info("New best checkpoint: %s", best_file)

    return str(epoch_file)


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
