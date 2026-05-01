"""
system1_vanilla/vanilla_llm.py
System 1 — Vanilla LLM Baseline
"""

import os
# Prevent transformers from importing TensorFlow/Flax if present in the environment.
# This project uses PyTorch only; TF can break on some Windows+NumPy setups.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import time
import torch
import psutil
import pynvml
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 128
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"🔄 Loading {MODEL_ID} in float16 on {DEVICE}...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"✅ Model loaded. GPU memory used: {allocated:.0f} MB")
    return tokenizer, model


def build_prompt(question: str) -> str:
    return (
        f"<|system|>\nYou are a helpful assistant. Answer questions accurately and concisely.</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )


def get_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except Exception:
        return torch.cuda.memory_allocated(0) / (1024 ** 2)


def get_cpu_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def generate_answer(question: str, tokenizer, model) -> dict:
    prompt  = build_prompt(question)
    inputs  = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    n_input = inputs["input_ids"].shape[1]

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
    **inputs,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    repetition_penalty=1.3,        # ADD THIS — stops looping
    pad_token_id=tokenizer.eos_token_id
)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency_ms    = (time.perf_counter() - start) * 1000
    new_token_ids = output_ids[0][n_input:]
    answer        = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    return {
        "question":         question,
        "answer":           answer,
        "latency_ms":       round(latency_ms, 2),
        "gpu_memory_mb":    round(get_gpu_memory_mb(), 2),
        "cpu_memory_mb":    round(get_cpu_memory_mb(), 2),
        "tokens_generated": len(new_token_ids)
    }


def run_pipeline(qa_pairs: list, tokenizer, model) -> list:
    from tqdm import tqdm
    results = []
    print(f"\n🚀 Running System 1 (Vanilla LLM) on {len(qa_pairs)} questions...")
    for qa in tqdm(qa_pairs):
        result = generate_answer(qa["question"], tokenizer, model)
        result["ground_truth"] = qa.get("ground_truth") or qa.get("answer", "")
        result["answer_aliases"] = qa.get("answer_aliases", [])
        result["system"]         = "vanilla"
        results.append(result)

    latencies = [r["latency_ms"] for r in results]
    throughput = len(results) / (sum(latencies) / 1000)
    print(f"\n📊 System 1 Summary:")
    print(f"   Avg latency : {sum(latencies)/len(latencies):.1f} ms")
    print(f"   Throughput  : {throughput:.2f} queries/sec")
    return results


if __name__ == "__main__":
    tokenizer, model = load_model()
    test_questions = [
        "Who invented the telephone?",
        "What is the capital of France?",
        "In what year did World War II end?"
    ]
    for q in test_questions:
        result = generate_answer(q, tokenizer, model)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"   Latency: {result['latency_ms']} ms | GPU: {result['gpu_memory_mb']} MB")