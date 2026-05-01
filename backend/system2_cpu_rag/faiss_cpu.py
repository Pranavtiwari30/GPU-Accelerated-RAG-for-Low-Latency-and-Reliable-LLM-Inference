# Your CPU RAG Faiss implementation will go here
"""
system2_cpu_rag/faiss_cpu.py
Builds and queries a FAISS CPU flat index.
"""

import os
import time
import numpy as np
import faiss

INDEX_PATH = "./data/indexes/faiss_cpu.index"
TOP_K      = 3


def build_index(embeddings: np.ndarray, save_path: str = INDEX_PATH) -> faiss.Index:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, save_path)
    print(f"✅ FAISS CPU index built: {index.ntotal} vectors | saved → {save_path}")
    return index


def load_index(path: str = INDEX_PATH) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS CPU index not found at {path}. Build it first.")
    index = faiss.read_index(path)
    print(f"✅ FAISS CPU index loaded: {index.ntotal} vectors")
    return index


def search(query_embedding: np.ndarray, index: faiss.Index, top_k: int = TOP_K):
    start = time.perf_counter()
    scores, indices = index.search(query_embedding, top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return scores, indices, elapsed_ms


def get_top_documents(query_embedding: np.ndarray, index: faiss.Index,
                      documents: list, top_k: int = TOP_K) -> tuple:
    scores, indices, latency_ms = search(query_embedding, index, top_k)
    retrieved = [documents[i] for i in indices[0] if i < len(documents)]
    return retrieved, latency_ms
