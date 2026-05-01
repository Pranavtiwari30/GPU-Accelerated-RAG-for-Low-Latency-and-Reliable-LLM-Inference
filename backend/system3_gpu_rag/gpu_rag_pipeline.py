"""
system3_gpu_rag/gpu_rag_pipeline.py

System 3 — GPU-Accelerated RAG Pipeline
Query → GPU Embedding → FAISS GPU → TinyLlama → Batched Inference → Output
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import psutil
import pynvml
import numpy as np

from system3_gpu_rag.embedder_gpu import load_embedder_gpu, embed_documents_gpu, embed_query_gpu
from system3_gpu_rag.faiss_gpu    import build_index_gpu, load_index_gpu, get_top_documents_gpu
from system1_vanilla.vanilla_llm  import load_model, get_gpu_memory_mb, DEVICE

# ── Index paths — one per dataset so caches never collide ─────────────────────
INDEX_PATH_TRIVIAQA = "./data/indexes/faiss_gpu_flat.index"
INDEX_PATH_SQUAD    = "./data/indexes/faiss_gpu_flat_squad.index"

MAX_NEW_TOKENS = 128
TOP_K          = 3
BATCH_SIZE     = 4


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


def setup_index_gpu(documents: list, embedder, force_rebuild: bool = False,
                    index_path: str = INDEX_PATH_TRIVIAQA):
    """Build or load the FAISS GPU index.

    Args:
        documents:     list of document dicts with 'text' key
        embedder:      GPU sentence-transformers embedder
        force_rebuild: always rebuild even if cache exists
        index_path:    path to save/load index
                       use INDEX_PATH_SQUAD for SQuAD to avoid cache collision
    """
    if not force_rebuild and os.path.exists(index_path):
        print(f"📂 Loading cached FAISS GPU index from {index_path}...")
        index = load_index_gpu(index_path)
        # Sanity check — warn if index size doesn't match documents
        if hasattr(index, 'ntotal') and index.ntotal != len(documents):
            print(f"⚠️  INDEX MISMATCH: {index.ntotal} vectors vs {len(documents)} documents.")
            print(f"   Rebuilding index to fix mismatch...")
            force_rebuild = True
        else:
            print(f"✅ Index loaded: {index.ntotal} vectors")
            return index

    print(f"🔨 Building FAISS GPU index ({len(documents)} docs) → {index_path}")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    doc_texts      = [d["text"] if isinstance(d, dict) else d for d in documents]
    doc_embeddings = embed_documents_gpu(doc_texts, embedder)
    index          = build_index_gpu(doc_embeddings, index_path)
    print(f"✅ FAISS GPU index built: {index.ntotal} vectors")
    return index


def generate_answer_gpu(question: str, tokenizer, model, embedder, index,
                         documents: list) -> dict:
    """Full GPU RAG pipeline for a single query."""
    t_start = time.perf_counter()

    # Step 1: GPU embedding
    query_emb, embed_lat_ms = embed_query_gpu(question, embedder)

    # Step 2: FAISS retrieval
    context_docs, ret_lat_ms = get_top_documents_gpu(query_emb, index, documents, TOP_K)

    # Step 3: Build prompt and tokenize
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
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
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
        "tokens_generated":       len(new_token_ids),
    }


def generate_batch_gpu(questions: list, tokenizer, model, embedder, index,
                        documents: list) -> list:
    """Process a batch of questions simultaneously for higher throughput."""
    t_start = time.perf_counter()
    results = []

    # Step 1: Batch embed all queries on GPU
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    query_embs = embedder.encode(
        questions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=DEVICE,
        batch_size=len(questions),
    ).astype(np.float32)
    if DEVICE == "cuda": torch.cuda.synchronize()
    embed_lat_ms = (time.perf_counter() - t0) * 1000

    # Step 2: Retrieve context for each query
    prompts       = []
    all_contexts  = []
    ret_lat_total = 0.0

    for i, question in enumerate(questions):
        context_docs, ret_ms = get_top_documents_gpu(query_embs[i:i+1], index, documents, TOP_K)
        all_contexts.append(context_docs)
        ret_lat_total += ret_ms
        prompts.append(build_prompt_with_context(question, context_docs))

    avg_ret_ms = ret_lat_total / len(questions)

    # Step 3: Batch tokenize with left padding
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(DEVICE)
    n_inputs = inputs["input_ids"].shape[1]

    # Step 4: Batched generation
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    if DEVICE == "cuda": torch.cuda.synchronize()
    gen_lat_ms   = (time.perf_counter() - t0) * 1000
    total_lat_ms = (time.perf_counter() - t_start) * 1000

    # Step 5: Decode all answers
    for i, question in enumerate(questions):
        new_ids = output_ids[i][n_inputs:]
        answer  = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append({
            "question":               question,
            "answer":                 answer,
            "retrieved_docs":         all_contexts[i],
            "embedding_latency_ms":   round(embed_lat_ms / len(questions), 2),
            "retrieval_latency_ms":   round(avg_ret_ms, 2),
            "generation_latency_ms":  round(gen_lat_ms / len(questions), 2),
            "latency_ms":             round(total_lat_ms / len(questions), 2),
            "gpu_memory_mb":          round(get_gpu_memory_mb(), 2),
            "cpu_memory_mb":          round(get_cpu_memory_mb(), 2),
            "tokens_generated":       len(new_ids),
        })

    return results


def run_pipeline(qa_pairs: list, documents: list, tokenizer, model,
                 use_batching: bool = True,
                 index_path: str = INDEX_PATH_TRIVIAQA,
                 force_rebuild: bool = False) -> list:
    """Run all QA pairs through the GPU RAG pipeline.

    Args:
        qa_pairs:      list of dicts with 'question', 'ground_truth', 'answer_aliases'
        documents:     list of document dicts with 'text' key
        tokenizer:     HuggingFace tokenizer
        model:         HuggingFace model
        use_batching:  process queries in batches of BATCH_SIZE for higher throughput
        index_path:    FAISS index path — pass INDEX_PATH_SQUAD for SQuAD evaluation
        force_rebuild: rebuild FAISS index even if cache exists
    """
    from tqdm import tqdm

    embedder = load_embedder_gpu()
    index    = setup_index_gpu(documents, embedder,
                               force_rebuild=force_rebuild,
                               index_path=index_path)

    results = []
    print(f"\n🚀 Running System 3 (GPU RAG) on {len(qa_pairs)} questions "
          f"[batching={'ON' if use_batching else 'OFF'}]...")

    if use_batching:
        for i in tqdm(range(0, len(qa_pairs), BATCH_SIZE)):
            batch     = qa_pairs[i:i+BATCH_SIZE]
            questions = [qa["question"] for qa in batch]
            batch_res = generate_batch_gpu(questions, tokenizer, model, embedder, index, documents)
            for j, result in enumerate(batch_res):
                result["ground_truth"]   = batch[j].get("ground_truth") or batch[j].get("answer", "")
                result["answer_aliases"] = batch[j].get("answer_aliases", [])
                result["system"]         = "gpu_rag"
                results.append(result)
    else:
        for qa in tqdm(qa_pairs):
            result = generate_answer_gpu(
                qa["question"], tokenizer, model, embedder, index, documents
            )
            result["ground_truth"]   = qa.get("ground_truth") or qa.get("answer", "")
            result["answer_aliases"] = qa.get("answer_aliases", [])
            result["system"]         = "gpu_rag"
            results.append(result)

    latencies  = [r["latency_ms"] for r in results]
    total_secs = sum(latencies) / 1000
    throughput = len(results) / total_secs if total_secs > 0 else 0

    print(f"\n📊 System 3 Summary:")
    print(f"   Avg total latency    : {sum(latencies)/len(latencies):.1f} ms")
    print(f"   Avg embed latency    : {sum(r['embedding_latency_ms'] for r in results)/len(results):.1f} ms")
    print(f"   Avg retrieval latency: {sum(r['retrieval_latency_ms'] for r in results)/len(results):.1f} ms")
    print(f"   Avg gen latency      : {sum(r['generation_latency_ms'] for r in results)/len(results):.1f} ms")
    print(f"   Throughput           : {throughput:.2f} queries/sec")
    print(f"   Avg GPU memory       : {sum(r['gpu_memory_mb'] for r in results)/len(results):.1f} MB")

    return results


if __name__ == "__main__":
    from data.load_dataset import get_data

    documents, qa_pairs = get_data()
    tokenizer, model    = load_model()

    test_qas = qa_pairs[:3]
    embedder = load_embedder_gpu()
    index    = setup_index_gpu(documents, embedder)

    for qa in test_qas:
        result = generate_answer_gpu(
            qa["question"], tokenizer, model, embedder, index, documents
        )
        print(f"\nQ: {result['question']}")
        print(f"A (predicted): {result['answer']}")
        print(f"A (truth):     {qa.get('ground_truth', qa.get('answer'))}")
        print(f"Total latency: {result['latency_ms']} ms")