"""
BERT vocabulary embedding generation.

Generates and saves BERT embeddings for the full shared vocabulary.
Uses the [CLS] token embedding from bert-base-uncased.

Output schema (saved as .npz):
    vocab:       list[str]   length V, word strings
    embeddings:  float32     (V, 768)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_bert():
    """
    Load bert-base-uncased tokenizer and model.

    Model is set to eval() and gradients are disabled during inference.

    Returns:
        Tuple of (tokenizer, model) — HuggingFace AutoTokenizer and AutoModel.
    """
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def get_bert_embedding(word: str, tokenizer, model) -> np.ndarray:
    """
    Get the [CLS] token embedding for a single word.

    Args:
        word: Input word string.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model in inference mode with no gradient tracking.

    Returns:
        float32 array of shape (768,).
    """
    import torch
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy().astype(np.float32)


def generate_vocab_embeddings(vocabulary: list[str], batch_size: int = 64) -> dict:
    """
    Generate BERT embeddings for all words in the vocabulary.

    Processes in batches for efficiency. Logs progress every 100 words.

    Args:
        vocabulary: Sorted list of word strings.
        batch_size: Words to process per forward pass.

    Returns:
        Dict with keys 'vocab' (list[str]) and 'embeddings' (float32 (V, 768)).
    """
    import torch
    tokenizer, model = load_bert()
    embeddings = []
    for i in range(0, len(vocabulary), batch_size):
        batch = vocabulary[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)
        embeddings.append(cls_embeddings)
        if (i + batch_size) % 100 == 0 or i == 0:
            logger.info(f"Generated embeddings for {min(i + batch_size, len(vocabulary))}/{len(vocabulary)} words")
    return {"vocab": vocabulary, "embeddings": np.vstack(embeddings).astype(np.float32)}


def save_vocab_embeddings(vocab_data: dict, output_path: str) -> str:
    """
    Save vocabulary embeddings to .npz.

    Validates that embeddings are non-empty and finite before saving.

    Args:
        vocab_data: Output of generate_vocab_embeddings().
        output_path: Path to save (e.g., 'data/processed/vocab_embeddings.npz').

    Returns:
        Path to saved .npz file.

    Raises:
        AssertionError: If embeddings contain NaN/Inf or vocab is empty.
    """
    raise NotImplementedError


def run(vocabulary: list[str], output_path: str) -> str:
    """
    End-to-end: load BERT, generate embeddings for vocabulary, save.

    Args:
        vocabulary: Sorted list of word strings (from align_splits.py).
        output_path: Output path for .npz file.

    Returns:
        Path to saved .npz file.
    """
    raise NotImplementedError
