# Your GPU RAG embedder implementation will go here
"""
system3_gpu_rag/embedder_gpu.py
GPU-accelerated embedding using all-MiniLM-L6-v2.
"""

import os
# Prevent transformers (pulled in by sentence-transformers) from importing TensorFlow/Flax.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256


def load_embedder_gpu():
    print(f"🔄 Loading embedding model: {MODEL_NAME} on {DEVICE.upper()}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"✅ GPU Embedder loaded on {DEVICE.upper()}.")
    return model


def embed_documents_gpu(texts: list, embedder) -> np.ndarray:
    print(f"🔨 Embedding {len(texts)} documents on {DEVICE.upper()}...")
    t0 = time.perf_counter()
    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=DEVICE
    )
    elapsed = time.perf_counter() - t0
    print(f"✅ Done: {embeddings.shape} in {elapsed:.2f}s ({len(texts)/elapsed:.1f} docs/sec)")
    return embeddings.astype(np.float32)


def embed_query_gpu(query: str, embedder) -> tuple:
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=DEVICE
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000
    return embedding.astype(np.float32), round(latency_ms, 2)