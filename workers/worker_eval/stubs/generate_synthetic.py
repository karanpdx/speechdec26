"""
Synthetic data generators for worker_eval development.

Generates realistic-looking (but random) arrays matching the exact shapes
and schemas that Person 3 will export from trained models.

Use these for all tests in tests/. No real models or data required.
"""

import numpy as np


def make_retrieval_inputs(
    n_samples: int = 200,
    vocab_size: int = 50,
    embed_dim: int = 768,
    n_subjects: int = 5,
    top1_accuracy: float = 0.3,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic retrieval inputs for Stage 1 evaluation.

    Simulates a model with approximately `top1_accuracy` retrieval accuracy
    by placing a fraction of neural embeddings near their correct text embedding.

    Args:
        n_samples: Number of test samples.
        vocab_size: Number of words in vocabulary.
        embed_dim: Embedding dimensionality.
        n_subjects: Number of test subjects.
        top1_accuracy: Approximate fraction of samples with correct top-1.
        seed: Random seed.

    Returns:
        Dict with keys:
            'neural_embeddings': float32 (n_samples, embed_dim) — L2 normalized
            'text_embeddings':   float32 (vocab_size, embed_dim) — L2 normalized
            'labels':            list[str] length n_samples
            'vocab':             list[str] length vocab_size
            'subject_ids':       list[str] length n_samples
    """
    rng = np.random.RandomState(seed)
    vocab = [f"word_{i:03d}" for i in range(vocab_size)]
    subjects = [f"sub-{i:02d}" for i in range(n_subjects)]

    # Ground truth text embeddings
    text_embeddings = rng.randn(vocab_size, embed_dim).astype(np.float32)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # Assign labels
    label_indices = rng.randint(0, vocab_size, size=n_samples)
    labels = [vocab[i] for i in label_indices]
    subject_ids = [subjects[i % n_subjects] for i in range(n_samples)]

    # Neural embeddings: fraction near correct embedding, rest random
    n_correct = int(n_samples * top1_accuracy)
    neural_embeddings = rng.randn(n_samples, embed_dim).astype(np.float32)

    # Make first n_correct samples close to their correct text embedding
    correct_indices = rng.choice(n_samples, size=n_correct, replace=False)
    for idx in correct_indices:
        noise = rng.randn(embed_dim).astype(np.float32) * 0.1
        neural_embeddings[idx] = text_embeddings[label_indices[idx]] + noise

    neural_embeddings /= np.linalg.norm(neural_embeddings, axis=1, keepdims=True)

    return {
        "neural_embeddings": neural_embeddings,
        "text_embeddings": text_embeddings,
        "labels": labels,
        "vocab": vocab,
        "subject_ids": subject_ids,
    }


def make_reconstruction_inputs(
    n_samples: int = 50,
    n_channels: int = 64,
    n_timepoints: int = 175,
    n_voxels: int = 1000,
    sfreq: float = 256.0,
    r_erp: float = 0.4,
    r_fmri: float = 0.3,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic reconstruction inputs for Stage 2 evaluation.

    Simulates predicted signals with approximately `r_erp` correlation
    to real EEG signals and `r_fmri` correlation to real fMRI beta maps.

    Args:
        n_samples: Number of test samples (words).
        n_channels: Number of EEG channels.
        n_timepoints: Time samples per epoch.
        n_voxels: Number of fMRI voxels.
        sfreq: Sampling frequency (Hz).
        r_erp: Approximate ERP correlation (0 = random, 1 = perfect).
        r_fmri: Approximate fMRI spatial correlation.
        seed: Random seed.

    Returns:
        Dict with keys:
            'pred_eeg':    float32 (n_samples, n_channels, n_timepoints)
            'real_eeg':    float32 (n_samples, n_channels, n_timepoints)
            'pred_fmri':   float32 (n_samples, n_voxels)
            'real_fmri':   float32 (n_samples, n_voxels)
            'voxel_coords': int32 (n_voxels, 3)
            'ch_names':    list[str] length n_channels
            'sfreq':       float
    """
    rng = np.random.RandomState(seed)

    real_eeg = rng.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    noise_eeg = rng.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    pred_eeg = (r_erp * real_eeg + (1 - r_erp) * noise_eeg).astype(np.float32)

    real_fmri = rng.randn(n_samples, n_voxels).astype(np.float32)
    noise_fmri = rng.randn(n_samples, n_voxels).astype(np.float32)
    pred_fmri = (r_fmri * real_fmri + (1 - r_fmri) * noise_fmri).astype(np.float32)

    voxel_coords = rng.randint(-90, 90, size=(n_voxels, 3)).astype(np.int32)
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]

    return {
        "pred_eeg": pred_eeg,
        "real_eeg": real_eeg,
        "pred_fmri": pred_fmri,
        "real_fmri": real_fmri,
        "voxel_coords": voxel_coords,
        "ch_names": ch_names,
        "sfreq": sfreq,
    }


def make_round_trip_inputs(
    n_words: int = 30,
    embed_dim: int = 768,
    round_trip_similarity: float = 0.6,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic round-trip consistency inputs.

    Args:
        n_words: Number of test words.
        embed_dim: Embedding dimensionality.
        round_trip_similarity: Target cosine similarity between original and round-trip.
        seed: Random seed.

    Returns:
        Dict with keys:
            'original_embeddings':   float32 (n_words, embed_dim) L2-normalized
            'round_trip_embeddings': float32 (n_words, embed_dim) L2-normalized
            'word_labels':           list[str] length n_words
    """
    rng = np.random.RandomState(seed)
    original = rng.randn(n_words, embed_dim).astype(np.float32)
    original /= np.linalg.norm(original, axis=1, keepdims=True)

    noise = rng.randn(n_words, embed_dim).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)

    # Orthogonalize noise against the original embedding so the resulting cosine
    # after normalization is approximately the requested round_trip_similarity.
    projection = np.sum(noise * original, axis=1, keepdims=True) * original
    orthogonal_noise = noise - projection
    orthogonal_noise /= np.linalg.norm(orthogonal_noise, axis=1, keepdims=True) + 1e-8

    mix = np.sqrt(max(0.0, 1.0 - round_trip_similarity ** 2))
    round_trip = round_trip_similarity * original + mix * orthogonal_noise
    round_trip /= np.linalg.norm(round_trip, axis=1, keepdims=True)

    return {
        "original_embeddings": original,
        "round_trip_embeddings": round_trip,
        "word_labels": [f"word_{i:03d}" for i in range(n_words)],
    }
