# Your CPU RAG embedder implementation will go here
"""
system2_cpu_rag/embedder.py

CPU-based embedding using all-MiniLM-L6-v2.
Used by System 2 (CPU RAG pipeline).
"""

import os
# Prevent transformers (pulled in by sentence-transformers) from importing TensorFlow/Flax.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_embedder():
    """Load MiniLM on CPU."""
    print(f"🔄 Loading embedding model: {MODEL_NAME} (CPU)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("✅ Embedder loaded on CPU.")
    return model


def embed_documents(texts: list, embedder) -> np.ndarray:
    """
    Embed a list of document strings on CPU.
    Returns np.ndarray of shape (N, 384), dtype float32
    """
    print(f"🔨 Embedding {len(texts)} documents on CPU...")
    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def embed_query(query: str, embedder) -> np.ndarray:
    """
    Embed a single query string on CPU.
    Returns np.ndarray of shape (1, 384), dtype float32
    """
    embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embedding.astype(np.float32)