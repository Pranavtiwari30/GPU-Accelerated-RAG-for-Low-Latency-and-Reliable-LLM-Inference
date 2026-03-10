"""
system2_cpu_rag/cpu_rag_pipeline.py

System 2 — CPU RAG Pipeline
Query → CPU Embedding → FAISS CPU Retrieval → Context Augmentation → TinyLlama Generation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import psutil
import pynvml

from system2_cpu_rag.embedder  import load_embedder, embed_documents, embed_query
from system2_cpu_rag.faiss_cpu import build_index, load_index, get_top_documents
from system1_vanilla.vanilla_llm import load_model, get_gpu_memory_mb, DEVICE

INDEX_PATH     = "./data/indexes/faiss_cpu.index"
MAX_NEW_TOKENS = 128
TOP_K          = 3


def get_cpu_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def build_prompt_with_context(question: str, context_docs: list) -> str:
    """TinyLlama chat format with retrieved context."""
    context = "\n\n".join(
        [f"[Document {i+1}]: {doc}" for i, doc in enumerate(context_docs)]
    )
    return (
        f"<|system|>\nYou are a helpful assistant. Use the provided documents to answer accurately and concisely.</s>\n"
        f"<|user|>\nDocuments:\n{context}\n\nQuestion: {question}</s>\n"
        f"<|assistant|>\n"
    )


def setup_index(documents: list, embedder, force_rebuild: bool = False):
    """Build FAISS CPU index from documents (or load cached version)."""
    if not force_rebuild and os.path.exists(INDEX_PATH):
        print("📂 Loading cached FAISS CPU index...")
        return load_index(INDEX_PATH)
    print("🔨 Building FAISS CPU index from scratch...")
    doc_embeddings = embed_documents(documents, embedder)
    index          = build_index(doc_embeddings, INDEX_PATH)
    return index


def generate_answer_with_context(question: str, tokenizer, model, embedder, index, documents: list) -> dict:
    """Full CPU RAG pipeline for a single query."""
    t_start = time.perf_counter()

    # Step 1: Embed query (CPU)
    t0           = time.perf_counter()
    query_emb    = embed_query(question, embedder)
    embed_lat_ms = (time.perf_counter() - t0) * 1000

    # Step 2: Retrieve top-k docs (FAISS CPU)
    context_docs, ret_lat_ms = get_top_documents(query_emb, index, documents, TOP_K)

    # Step 3: Build augmented prompt
    prompt = build_prompt_with_context(question, context_docs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    n_input = inputs["input_ids"].shape[1]

    # Step 4: Generate
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
    **inputs,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    repetition_penalty=1.3,        # ADD THIS — stops looping
    pad_token_id=tokenizer.eos_token_id
)

    if DEVICE == "cuda": torch.cuda.synchronize()
    gen_lat_ms   = (time.perf_counter() - t0) * 1000
    total_lat_ms = (time.perf_counter() - t_start) * 1000

    new_token_ids = output_ids[0][n_input:]
    answer        = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    return {
        "question":               question,
        "answer":                 answer,
        "retrieved_docs":         context_docs,
        "embedding_latency_ms":   round(embed_lat_ms, 2),
        "retrieval_latency_ms":   round(ret_lat_ms, 2),
        "generation_latency_ms":  round(gen_lat_ms, 2),
        "latency_ms":             round(total_lat_ms, 2),
        "gpu_memory_mb":          round(get_gpu_memory_mb(), 2),
        "cpu_memory_mb":          round(get_cpu_memory_mb(), 2),
        "tokens_generated":       len(new_token_ids)
    }


def run_pipeline(qa_pairs: list, documents: list, tokenizer, model) -> list:
    """Run all QA pairs through the CPU RAG pipeline."""
    from tqdm import tqdm

    embedder = load_embedder()
    index    = setup_index(documents, embedder, force_rebuild=False)

    results = []
    print(f"\n🚀 Running System 2 (CPU RAG) on {len(qa_pairs)} questions...")

    for qa in tqdm(qa_pairs):
        result = generate_answer_with_context(
            qa["question"], tokenizer, model, embedder, index, documents
        )
        result["ground_truth"]   = qa["answer"]
        result["answer_aliases"] = qa.get("answer_aliases", [])
        result["system"]         = "cpu_rag"
        results.append(result)

    latencies  = [r["latency_ms"] for r in results]
    total_secs = sum(latencies) / 1000
    throughput = len(results) / total_secs if total_secs > 0 else 0

    print(f"\n📊 System 2 Summary:")
    print(f"   Avg total latency    : {sum(latencies)/len(latencies):.1f} ms")
    print(f"   Avg embed latency    : {sum(r['embedding_latency_ms'] for r in results)/len(results):.1f} ms")
    print(f"   Avg retrieval latency: {sum(r['retrieval_latency_ms'] for r in results)/len(results):.1f} ms")
    print(f"   Avg gen latency      : {sum(r['generation_latency_ms'] for r in results)/len(results):.1f} ms")
    print(f"   Throughput           : {throughput:.2f} queries/sec")

    return results


if __name__ == "__main__":
    from data.load_dataset import get_data

    documents, qa_pairs = get_data()
    tokenizer, model    = load_model()

    test_qas = qa_pairs[:3]
    embedder = load_embedder()
    index    = setup_index(documents, embedder)

    for qa in test_qas:
        result = generate_answer_with_context(
            qa["question"], tokenizer, model, embedder, index, documents
        )
        print(f"\nQ: {result['question']}")
        print(f"A (predicted): {result['answer']}")
        print(f"A (truth):     {qa['answer']}")
        print(f"Total latency: {result['latency_ms']} ms")