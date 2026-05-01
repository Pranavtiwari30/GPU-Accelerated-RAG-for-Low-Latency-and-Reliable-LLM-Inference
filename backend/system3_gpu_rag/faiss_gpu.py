# Your GPU RAG Faiss implementation will go here
"""
system3_gpu_rag/faiss_gpu.py
FAISS GPU index with CPU fallback for Windows.
"""

import os
import time
import numpy as np
import faiss

INDEX_PATH = "./data/indexes/faiss_gpu_flat.index"
TOP_K      = 3


def _get_gpu_resource():
    try:
        res = faiss.StandardGpuResources()
        print("✅ FAISS GPU resources initialized.")
        return res
    except AttributeError:
        print("⚠️  faiss-gpu not available — falling back to CPU FAISS.")
        return None


def build_index_gpu(embeddings: np.ndarray, save_path: str = INDEX_PATH) -> faiss.Index:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dim = embeddings.shape[1]
    res = _get_gpu_resource()

    if res is not None:
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        index_gpu = faiss.GpuIndexFlatIP(res, dim, flat_config)
        index_gpu.add(embeddings)
        index_cpu = faiss.index_gpu_to_cpu(index_gpu)
        faiss.write_index(index_cpu, save_path)
        print(f"✅ FAISS GPU index built: {index_gpu.ntotal} vectors | saved → {save_path}")
        return index_gpu
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, save_path)
        print(f"✅ FAISS CPU fallback index built: {index.ntotal} vectors | saved → {save_path}")
        return index


def load_index_gpu(path: str = INDEX_PATH) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index not found at {path}. Build it first.")
    index_cpu = faiss.read_index(path)
    res = _get_gpu_resource()
    if res is not None:
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        print(f"✅ Index moved to GPU: {index_gpu.ntotal} vectors")
        return index_gpu
    else:
        print(f"✅ Index loaded on CPU fallback: {index_cpu.ntotal} vectors")
        return index_cpu


def search_gpu(query_embedding: np.ndarray, index: faiss.Index, top_k: int = TOP_K):
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.synchronize()
    except ImportError:
        pass
    t0 = time.perf_counter()
    scores, indices = index.search(query_embedding, top_k)
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.synchronize()
    except ImportError:
        pass
    latency_ms = (time.perf_counter() - t0) * 1000
    return scores, indices, round(latency_ms, 2)


def get_top_documents_gpu(query_embedding: np.ndarray, index: faiss.Index,
                          documents: list, top_k: int = TOP_K) -> tuple:
    scores, indices, latency_ms = search_gpu(query_embedding, index, top_k)
    retrieved = [documents[i] for i in indices[0] if i < len(documents)]
    return retrieved, latency_ms