"""
Stage 2 training loop — decoder training with frozen Stage 1 encoders.

Loads a Stage 1 checkpoint, freezes all encoder weights, then trains
modality-specific decoders to reconstruct neural signals from shared embeddings.

CRITICAL: Encoder weights are frozen and verified before any gradient step.

Entry point: scripts/train_stage2.py
"""

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def load_stage1_checkpoint(checkpoint_path: str, models: dict, device: str) -> dict:
    """
    Load Stage 1 checkpoint and freeze all encoder weights.

    Verifies that all encoder parameters have requires_grad=False after loading.

    Args:
        checkpoint_path: Path to Stage 1 best.pt checkpoint.
        models: Dict with Stage 1 model keys from build_models().
        device: torch device string.

    Returns:
        Dict of loaded encoder modules with frozen weights.

    Raises:
        AssertionError: If any encoder parameter still has requires_grad=True.
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If the checkpoint is missing model state dicts.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dicts" not in checkpoint:
        raise KeyError("Stage 1 checkpoint missing 'model_state_dicts'")

    state_dicts = checkpoint["model_state_dicts"]
    required_keys = ("eeg_encoder", "meg_encoder", "fmri_encoder", "projector")
    missing = [key for key in required_keys if key not in state_dicts]
    if missing:
        raise KeyError(f"Stage 1 checkpoint missing model states for: {missing}")

    for key, model in models.items():
        if key in state_dicts:
            model.load_state_dict(state_dicts[key])
            logger.info("load_stage1_checkpoint: loaded state for %s", key)

    for key in ("eeg_encoder", "meg_encoder", "fmri_encoder", "projector"):
        model = models[key]
        model.train(False)
        for param in model.parameters():
            param.requires_grad = False

    models["subject_emb"].train(True)
    for param in models["subject_emb"].parameters():
        param.requires_grad = True

    encoders = {
        "eeg": models["eeg_encoder"],
        "meg": models["meg_encoder"],
        "fmri": models["fmri_encoder"],
    }
    verify_encoders_frozen(encoders)
    return encoders


def verify_encoders_frozen(encoders: dict) -> None:
    """
    Assert that all encoder parameters have requires_grad=False.

    Call this before every training step in Stage 2.

    Args:
        encoders: Dict mapping modality name to encoder nn.Module.

    Raises:
        AssertionError: If any parameter requires grad.
    """
    for name, encoder in encoders.items():
        for param_name, param in encoder.named_parameters():
            if param.requires_grad:
                raise AssertionError(
                    f"Encoder '{name}' parameter '{param_name}' is still trainable during Stage 2"
                )


def freq_domain_loss(pred: torch.Tensor, target: torch.Tensor, sfreq: float) -> torch.Tensor:
    """
    Frequency-domain loss penalizing PSD differences across neural frequency bands.

    Bands: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz).
    Bands are weighted equally. Computed on the time dimension (dim=-1).

    Args:
        pred:   (batch, n_channels, n_timepoints) predicted signal
        target: (batch, n_channels, n_timepoints) real signal
        sfreq:  sampling frequency in Hz

    Returns:
        Scalar loss (mean PSD MSE across all bands and channels).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must match shape, got {pred.shape} vs {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"freq_domain_loss expects 3D tensors, got shape {pred.shape}")

    n_timepoints = pred.shape[-1]
    freqs = torch.fft.rfftfreq(n_timepoints, d=1.0 / float(sfreq)).to(pred.device)
    pred_psd = torch.fft.rfft(pred, dim=-1).abs().pow(2)
    target_psd = torch.fft.rfft(target, dim=-1).abs().pow(2)

    bands = (
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
    )

    band_losses = []
    for low, high in bands:
        band_mask = (freqs >= low) & (freqs < high)
        if not torch.any(band_mask):
            continue
        pred_band = pred_psd[..., band_mask].mean(dim=-1)
        target_band = target_psd[..., band_mask].mean(dim=-1)
        band_losses.append(F.mse_loss(pred_band, target_band))

    if not band_losses:
        return pred.sum() * 0.0

    return torch.stack(band_losses).mean()


def spatial_smoothness_loss(
    pred: torch.Tensor,
    adjacency_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial smoothness loss for fMRI: penalize L2 norm of differences
    between adjacent voxels in predicted beta maps.

    Args:
        pred: (batch, n_voxels) predicted beta maps
        adjacency_indices: (n_edges, 2) long tensor of adjacent voxel index pairs,
                           precomputed from voxel_coords at startup.

    Returns:
        Scalar loss.
    """
    if pred.ndim != 2:
        raise ValueError(f"spatial_smoothness_loss expects 2D tensor, got shape {pred.shape}")
    if adjacency_indices is None or adjacency_indices.numel() == 0:
        return pred.sum() * 0.0

    edge_index = adjacency_indices.to(pred.device)
    left = pred.index_select(1, edge_index[:, 0])
    right = pred.index_select(1, edge_index[:, 1])
    return (left - right).pow(2).mean()


def build_voxel_adjacency(voxel_coords: torch.Tensor, max_dist_mm: float = 6.0) -> torch.Tensor:
    """
    Build a sparse adjacency tensor from MNI voxel coordinates.

    Two voxels are adjacent if their Euclidean distance in MNI space <= max_dist_mm.

    Computed once at startup and reused for all batches.

    Args:
        voxel_coords: int32 (n_voxels, 3) MNI coordinates.
        max_dist_mm: Maximum distance to consider voxels adjacent.

    Returns:
        (n_edges, 2) long tensor of adjacent voxel index pairs.
    """
    coords = voxel_coords.float()
    n_voxels = coords.shape[0]
    if n_voxels < 2:
        return torch.empty((0, 2), dtype=torch.long)

    chunk_size = 512
    all_indices = torch.arange(n_voxels, dtype=torch.long)
    edge_chunks = []

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        dist = torch.cdist(coords[start:end], coords)
        for row_offset in range(end - start):
            src_idx = start + row_offset
            valid = (dist[row_offset] <= max_dist_mm) & (all_indices > src_idx)
            dst_idx = all_indices[valid]
            if dst_idx.numel() == 0:
                continue
            src_col = torch.full((dst_idx.numel(), 1), src_idx, dtype=torch.long)
            edge_chunks.append(torch.cat((src_col, dst_idx.unsqueeze(1)), dim=1))

    if not edge_chunks:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.cat(edge_chunks, dim=0)


def build_decoders(config: dict) -> dict:
    """
    Instantiate Stage 2 decoders.

    Args:
        config: Parsed train_stage2.yaml.

    Returns:
        Dict: {'eeg_decoder', 'meg_decoder', 'fmri_decoder'}
    """
    try:
        from src.models.decoders import EEGDecoder, MEGDecoder, fMRIDecoder
        logger.info("build_decoders: using src.models.decoders")
    except ImportError:
        logger.warning("build_decoders: src.models.decoders not available — using stubs/model_stubs.py")
        from stubs.model_stubs import EEGDecoder, MEGDecoder, fMRIDecoder

    device = torch.device(config.get("device", "cpu"))
    decoders = {
        "eeg_decoder": EEGDecoder(
            embed_dim=config["embed_dim"],
            subject_embed_dim=config["subject_embed_dim"],
            n_channels=config["eeg_channels"],
            n_timepoints=config["eeg_timepoints"],
        ).to(device),
        "meg_decoder": MEGDecoder(
            embed_dim=config["embed_dim"],
            subject_embed_dim=config["subject_embed_dim"],
            n_channels=config["meg_channels"],
            n_timepoints=config["meg_timepoints"],
        ).to(device),
        "fmri_decoder": fMRIDecoder(
            embed_dim=config["embed_dim"],
            subject_embed_dim=config["subject_embed_dim"],
            n_voxels=config["fmri_voxels"],
        ).to(device),
    }
    return decoders


def _iter_trainable_params(decoders: dict, subject_emb) -> list[torch.nn.Parameter]:
    params = []
    for decoder in decoders.values():
        params.extend(list(decoder.parameters()))
    params.extend(list(subject_emb.parameters()))
    return params


def _run_epoch(
    encoders: dict,
    decoders: dict,
    projector,
    subject_emb,
    dataloader,
    config: dict,
    optimizer: torch.optim.Optimizer | None,
    adjacency_indices: torch.Tensor | None,
    training: bool,
) -> dict:
    device = torch.device(config.get("device", "cpu"))
    mse_weight = float(config.get("mse_weight", 1.0))
    freq_weight = float(config.get("freq_domain_weight", 0.1))
    smooth_weight = float(config.get("spatial_smooth_weight", 0.05))
    log_every = int(config.get("log_every_n_steps", 10))
    grad_clip = float(config.get("grad_clip", 1.0))

    verify_encoders_frozen(encoders)
    projector.train(False)
    for encoder in encoders.values():
        encoder.train(False)
    subject_emb.train(training)
    for decoder in decoders.values():
        decoder.train(training)

    trainable_params = _iter_trainable_params(decoders, subject_emb)
    running = {
        "total": 0.0,
        "mse": 0.0,
        "freq": 0.0,
        "smooth": 0.0,
        "eeg": 0.0,
        "meg": 0.0,
        "fmri": 0.0,
    }
    n_steps = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(dataloader):
            verify_encoders_frozen(encoders)
            if training:
                optimizer.zero_grad()

            total_mse = torch.tensor(0.0, device=device)
            total_freq = torch.tensor(0.0, device=device)
            total_smooth = torch.tensor(0.0, device=device)
            modality_loss_totals = {
                "eeg": torch.tensor(0.0, device=device),
                "meg": torch.tensor(0.0, device=device),
                "fmri": torch.tensor(0.0, device=device),
            }

            for modality in ("eeg", "meg", "fmri"):
                if modality not in batch:
                    continue

                modal_data = batch[modality]
                data = modal_data["data"].to(device)
                subject_ids = modal_data["subject_idx"].to(device)

                with torch.no_grad():
                    shared_emb = encoders[modality](data)

                subj_emb = subject_emb(subject_ids)
                decoder_key = f"{modality}_decoder"
                recon = decoders[decoder_key](shared_emb, subj_emb)

                logger.info(
                    "_run_epoch[%s]: input_shape=%s shared_shape=%s recon_shape=%s",
                    modality,
                    tuple(data.shape),
                    tuple(shared_emb.shape),
                    tuple(recon.shape),
                )

                mse_loss = F.mse_loss(recon, data)
                total_mse = total_mse + mse_loss
                modality_total = mse_weight * mse_loss

                if modality in ("eeg", "meg"):
                    sfreq = float(modal_data.get("sfreq", config.get("target_sfreq", 256.0)))
                    band_loss = freq_domain_loss(recon, data, sfreq=sfreq)
                    total_freq = total_freq + band_loss
                    modality_total = modality_total + (freq_weight * band_loss)
                else:
                    band_loss = None

                if modality == "fmri":
                    smooth_loss = spatial_smoothness_loss(recon, adjacency_indices)
                    total_smooth = total_smooth + smooth_loss
                    modality_total = modality_total + (smooth_weight * smooth_loss)
                else:
                    smooth_loss = None

                modality_loss_totals[modality] = modality_total

            total_loss = (
                mse_weight * total_mse
                + freq_weight * total_freq
                + smooth_weight * total_smooth
            )

            if training and any(param.requires_grad for param in trainable_params):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "_run_epoch step=%d training=%s total=%.6f mse=%.6f freq=%.6f smooth=%.6f",
                    step,
                    training,
                    total_loss.item(),
                    total_mse.item(),
                    total_freq.item(),
                    total_smooth.item(),
                )

            running["total"] += total_loss.item()
            running["mse"] += total_mse.item()
            running["freq"] += total_freq.item()
            running["smooth"] += total_smooth.item()
            running["eeg"] += modality_loss_totals["eeg"].item()
            running["meg"] += modality_loss_totals["meg"].item()
            running["fmri"] += modality_loss_totals["fmri"].item()
            n_steps += 1

    denom = max(n_steps, 1)
    return {key: value / denom for key, value in running.items()}


def train_one_epoch(
    encoders: dict,
    decoders: dict,
    projector,
    subject_emb,
    dataloader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    adjacency_indices: torch.Tensor = None,
) -> dict:
    """
    Run one full Stage 2 training epoch.

    For each batch:
        1. Forward frozen encoder → shared embedding
        2. Forward decoder → predicted signal
        3. MSE loss between predicted and real signal
        4. Frequency-domain loss for EEG/MEG
        5. Spatial smoothness loss for fMRI
        6. Total = mse_weight * MSE + freq_weight * freq + smooth_weight * smooth
        7. Backward (decoder params only), optimizer step

    Verifies encoders_frozen before step 1.

    Args:
        encoders: Frozen encoder dict from load_stage1_checkpoint().
        decoders: Output of build_decoders().
        projector: Frozen SharedEmbeddingProjector.
        subject_emb: SubjectEmbedding (trainable for Stage 2).
        dataloader: Training DataLoader.
        optimizer: Optimizer over decoder parameters only.
        config: Parsed config dict.
        adjacency_indices: Precomputed voxel adjacency for spatial smoothness loss.

    Returns:
        Dict of mean epoch losses per modality.
    """
    return _run_epoch(
        encoders=encoders,
        decoders=decoders,
        projector=projector,
        subject_emb=subject_emb,
        dataloader=dataloader,
        config=config,
        optimizer=optimizer,
        adjacency_indices=adjacency_indices,
        training=True,
    )


def _save_checkpoint(
    encoders: dict,
    decoders: dict,
    projector,
    subject_emb,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
    is_best: bool = False,
) -> str:
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dicts": {
            "eeg_encoder": encoders["eeg"].state_dict(),
            "meg_encoder": encoders["meg"].state_dict(),
            "fmri_encoder": encoders["fmri"].state_dict(),
            "projector": projector.state_dict(),
            "subject_emb": subject_emb.state_dict(),
            "eeg_decoder": decoders["eeg_decoder"].state_dict(),
            "meg_decoder": decoders["meg_decoder"].state_dict(),
            "fmri_decoder": decoders["fmri_decoder"].state_dict(),
        },
    }

    epoch_file = ckpt_path / f"epoch_{epoch}.pt"
    torch.save(state, epoch_file)
    logger.info("_save_checkpoint: saved %s", epoch_file)

    if is_best:
        best_file = ckpt_path / "best.pt"
        torch.save(state, best_file)
        logger.info("_save_checkpoint: updated %s", best_file)

    return str(epoch_file)


def _build_adjacency_from_dataset(dataset, config: dict) -> torch.Tensor | None:
    for modality, path, _ in getattr(dataset, "_index", []):
        if modality != "fmri":
            continue
        npz = np.load(path, allow_pickle=False)
        if "voxel_coords" not in npz:
            logger.warning("_build_adjacency_from_dataset: voxel_coords missing in %s", path)
            return None
        voxel_coords = torch.from_numpy(npz["voxel_coords"].copy())
        adjacency = build_voxel_adjacency(
            voxel_coords,
            max_dist_mm=float(config.get("spatial_smooth_max_dist_mm", 6.0)),
        )
        logger.info(
            "_build_adjacency_from_dataset: voxel_coords_shape=%s adjacency_shape=%s",
            tuple(voxel_coords.shape),
            tuple(adjacency.shape),
        )
        return adjacency
    return None


def train(config_path: str) -> None:
    """
    Full Stage 2 training run.

    Loads Stage 1 checkpoint, freezes encoders, trains decoders.
    Does NOT update any encoder weights at any point.

    Args:
        config_path: Path to configs/train_stage2.yaml.
    """
    import yaml
    from torch.utils.data import DataLoader

    from src.training.dataset import MultiModalDataset, collate_fn
    from src.training.train_stage1 import build_models

    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)

    device = config.get("device", "cpu")
    config["device"] = device
    modalities = config.get("modalities", ["eeg", "meg", "fmri"])
    processed_base_dir = config.get("processed_base_dir", "data/processed")

    stage1_models = build_models(config)
    encoders = load_stage1_checkpoint(
        checkpoint_path=config["stage1_checkpoint"],
        models=stage1_models,
        device=device,
    )
    projector = stage1_models["projector"]
    subject_emb = stage1_models["subject_emb"].to(device)
    decoders = build_decoders(config)

    train_dataset = MultiModalDataset(
        split_json_path=config["split_file"],
        vocab_embeddings_path=config["vocab_embeddings"],
        processed_base_dir=processed_base_dir,
        split="train",
        modalities=modalities,
    )
    val_dataset = MultiModalDataset(
        split_json_path=config["split_file"],
        vocab_embeddings_path=config["vocab_embeddings"],
        processed_base_dir=processed_base_dir,
        split="val",
        modalities=modalities,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    adjacency_indices = _build_adjacency_from_dataset(train_dataset, config)
    optimizer = torch.optim.AdamW(
        _iter_trainable_params(decoders, subject_emb),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints/stage2/")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(checkpoint_dir) / "train_log.csv"
    fieldnames = [
        "epoch",
        "split",
        "total_loss",
        "mse_loss",
        "freq_loss",
        "smooth_loss",
        "eeg_loss",
        "meg_loss",
        "fmri_loss",
    ]

    n_epochs = int(config.get("n_epochs", 30))
    val_every = int(config.get("val_every_n_epochs", 5))
    best_val_total = float("inf")

    with open(log_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(n_epochs):
            logger.info("Stage 2 epoch %d/%d", epoch + 1, n_epochs)

            train_metrics = train_one_epoch(
                encoders=encoders,
                decoders=decoders,
                projector=projector,
                subject_emb=subject_emb,
                dataloader=train_loader,
                optimizer=optimizer,
                config=config,
                adjacency_indices=adjacency_indices,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "split": "train",
                    "total_loss": train_metrics["total"],
                    "mse_loss": train_metrics["mse"],
                    "freq_loss": train_metrics["freq"],
                    "smooth_loss": train_metrics["smooth"],
                    "eeg_loss": train_metrics["eeg"],
                    "meg_loss": train_metrics["meg"],
                    "fmri_loss": train_metrics["fmri"],
                }
            )
            csv_file.flush()

            val_metrics = {}
            if (epoch + 1) % val_every == 0 or epoch == n_epochs - 1:
                val_metrics = _run_epoch(
                    encoders=encoders,
                    decoders=decoders,
                    projector=projector,
                    subject_emb=subject_emb,
                    dataloader=val_loader,
                    config=config,
                    optimizer=None,
                    adjacency_indices=adjacency_indices,
                    training=False,
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "split": "val",
                        "total_loss": val_metrics["total"],
                        "mse_loss": val_metrics["mse"],
                        "freq_loss": val_metrics["freq"],
                        "smooth_loss": val_metrics["smooth"],
                        "eeg_loss": val_metrics["eeg"],
                        "meg_loss": val_metrics["meg"],
                        "fmri_loss": val_metrics["fmri"],
                    }
                )
                csv_file.flush()

            current_val_total = val_metrics.get("total", float("inf"))
            is_best = current_val_total < best_val_total
            if is_best:
                best_val_total = current_val_total

            if (epoch + 1) % 10 == 0 or is_best or epoch == n_epochs - 1:
                _save_checkpoint(
                    encoders=encoders,
                    decoders=decoders,
                    projector=projector,
                    subject_emb=subject_emb,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"train": train_metrics, "val": val_metrics},
                    checkpoint_dir=checkpoint_dir,
                    is_best=is_best,
                )

    logger.info("Stage 2 training complete. Best val total loss: %.6f", best_val_total)
