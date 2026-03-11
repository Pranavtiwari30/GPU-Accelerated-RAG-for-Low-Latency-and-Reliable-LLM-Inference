# Your evaluation metrics code will go here
"""
evaluation/metrics.py

All evaluation metrics for the project.

A. Performance Metrics
   - average_latency()
   - throughput()
   - gpu_memory_usage()
   - cpu_memory_usage()

B. Reliability Metrics
   - hallucination_rate()
   - factual_consistency_score()
   - answer_grounding_quality()
"""

import re
import string
import numpy as np
from typing import List, Dict


# ══════════════════════════════════════════════════════════════════════════════
# A. PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def average_latency(results: List[Dict]) -> float:
    """Average end-to-end latency in ms."""
    return np.mean([r["latency_ms"] for r in results])


def latency_p50_p95_p99(results: List[Dict]) -> Dict:
    """Latency percentiles — useful for understanding tail latency."""
    lats = sorted([r["latency_ms"] for r in results])
    n    = len(lats)
    return {
        "p50_ms": lats[int(n * 0.50)],
        "p95_ms": lats[int(n * 0.95)],
        "p99_ms": lats[min(int(n * 0.99), n-1)],
    }


def throughput(results: List[Dict]) -> float:
    """
    Queries per second.
    Uses the sum of individual latencies as total wall time estimate.
    """
    total_seconds = sum(r["latency_ms"] for r in results) / 1000
    return len(results) / total_seconds if total_seconds > 0 else 0.0


def avg_gpu_memory_mb(results: List[Dict]) -> float:
    """Average GPU memory usage in MB across all queries."""
    vals = [r["gpu_memory_mb"] for r in results if r.get("gpu_memory_mb", 0) > 0]
    return np.mean(vals) if vals else 0.0


def avg_cpu_memory_mb(results: List[Dict]) -> float:
    """Average CPU (RAM) usage in MB across all queries."""
    vals = [r["cpu_memory_mb"] for r in results]
    return np.mean(vals) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# B. RELIABILITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_answer(text: str) -> str:
    """
    Lowercase, remove punctuation, remove articles, collapse whitespace.
    Standard normalization from SQuAD evaluation script.
    """
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def is_correct(predicted: str, ground_truth: str, aliases: List[str]) -> bool:
    """
    Check if predicted answer is correct.
    Uses contains-match (more lenient than exact match) since LLMs often
    produce full sentences rather than bare answers.

    A prediction is correct if:
      - the normalized ground truth appears in the normalized prediction, OR
      - any normalized alias appears in the normalized prediction
    """
    pred_norm = normalize_answer(predicted)

    candidates = [ground_truth] + (aliases or [])
    for candidate in candidates:
        cand_norm = normalize_answer(candidate)
        if cand_norm and cand_norm in pred_norm:
            return True

    return False


def hallucination_rate(results: List[Dict]) -> float:
    """
    Hallucination Rate = Incorrect answers / Total answers

    An answer is "incorrect" (hallucinated) if it fails is_correct().
    Returns a float in [0, 1].
    """
    if not results:
        return 0.0

    incorrect = sum(
        1 for r in results
        if not is_correct(r["answer"], r["ground_truth"], r.get("answer_aliases", []))
    )
    return incorrect / len(results)


def factual_consistency_score(results: List[Dict]) -> float:
    """
    Factual Consistency = Correct answers / Total answers = 1 - hallucination_rate
    Returns a float in [0, 1].
    """
    return 1.0 - hallucination_rate(results)


def answer_grounding_quality(results: List[Dict]) -> float:
    """
    Answer Grounding Quality — only meaningful for RAG systems (Systems 2 & 3).

    Measures what fraction of correct answers can be traced back to the
    retrieved context (i.e., the answer string appears in at least one
    retrieved document).

    For vanilla LLM (no retrieved_docs), returns None.

    Returns float in [0, 1] or None.
    """
    if not results or "retrieved_docs" not in results[0]:
        return None

    grounded_correct = 0
    total_correct    = 0

    for r in results:
        if is_correct(r["answer"], r["ground_truth"], r.get("answer_aliases", [])):
            total_correct += 1
            # Check if any retrieved doc contains the ground truth
            gt_norm   = normalize_answer(r["ground_truth"])
            docs_text = " ".join(
    normalize_answer(d["text"] if isinstance(d, dict) else d)
    for d in r.get("retrieved_docs", [])
)
            if gt_norm and gt_norm in docs_text:
                grounded_correct += 1

    return grounded_correct / total_correct if total_correct > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(results: List[Dict], system_name: str) -> Dict:
    """
    Compute all metrics for a system's results and return a summary dict.
    """
    grounding = answer_grounding_quality(results)
    lat_pcts  = latency_p50_p95_p99(results)

    metrics = {
        "system":                    system_name,
        "n_queries":                 len(results),

        # Performance
        "avg_latency_ms":            round(average_latency(results), 2),
        "p50_latency_ms":            round(lat_pcts["p50_ms"], 2),
        "p95_latency_ms":            round(lat_pcts["p95_ms"], 2),
        "p99_latency_ms":            round(lat_pcts["p99_ms"], 2),
        "throughput_qps":            round(throughput(results), 3),
        "avg_gpu_memory_mb":         round(avg_gpu_memory_mb(results), 1),
        "avg_cpu_memory_mb":         round(avg_cpu_memory_mb(results), 1),

        # Reliability
        "hallucination_rate":        round(hallucination_rate(results), 4),
        "factual_consistency_score": round(factual_consistency_score(results), 4),
        "answer_grounding_quality":  round(grounding, 4) if grounding is not None else "N/A",
    }

    # Latency breakdown (RAG systems only)
    if "embedding_latency_ms" in results[0]:
        metrics["avg_embed_latency_ms"]    = round(np.mean([r["embedding_latency_ms"]   for r in results]), 2)
        metrics["avg_retrieval_latency_ms"]= round(np.mean([r["retrieval_latency_ms"]   for r in results]), 2)
        metrics["avg_gen_latency_ms"]      = round(np.mean([r["generation_latency_ms"]  for r in results]), 2)

    return metrics


def print_metrics_table(all_metrics: List[Dict]) -> None:
    """Pretty print comparison table of all 3 systems."""
    print("\n" + "="*80)
    print("📊  RESULTS COMPARISON TABLE")
    print("="*80)

    key_metrics = [
        ("avg_latency_ms",            "Avg Latency (ms)"),
        ("throughput_qps",            "Throughput (q/s)"),
        ("avg_gpu_memory_mb",         "GPU Memory (MB)"),
        ("hallucination_rate",        "Hallucination Rate"),
        ("factual_consistency_score", "Factual Consistency"),
        ("answer_grounding_quality",  "Answer Grounding"),
    ]

    # Header
    systems = [m["system"] for m in all_metrics]
    col_w   = 20
    print(f"{'Metric':<30}" + "".join(f"{s:>{col_w}}" for s in systems))
    print("-"*80)

    for key, label in key_metrics:
        row = f"{label:<30}"
        for m in all_metrics:
            val = m.get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)

    print("="*80 + "\n")
