"""
evaluation/run_eval.py

Master evaluation script. Runs all 3 systems and saves results.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --systems 1 3
    python evaluation/run_eval.py --n_questions 50
    python evaluation/run_eval.py --rebuild
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import pandas as pd
from datetime import datetime

from data.load_dataset                import get_data
from system1_vanilla.vanilla_llm      import load_model, run_pipeline as run_vanilla
from system2_cpu_rag.cpu_rag_pipeline import run_pipeline as run_cpu_rag
from system3_gpu_rag.gpu_rag_pipeline import run_pipeline as run_gpu_rag
from evaluation.metrics               import compute_all_metrics, print_metrics_table

RESULTS_DIR = "./results"


def save_results(results: list, system_name: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = os.path.join(RESULTS_DIR, f"{system_name}_{timestamp}.json")
    slim = [{k: v for k, v in r.items() if k != "retrieved_docs"} for r in results]
    with open(path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"💾 Results saved → {path}")
    return path


def save_metrics_csv(all_metrics: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = os.path.join(RESULTS_DIR, f"metrics_comparison_{timestamp}.csv")
    pd.DataFrame(all_metrics).to_csv(path, index=False)
    print(f"💾 Metrics CSV saved → {path}")
    return path


def main(systems=(1, 2, 3), n_questions=500, rebuild=False):
    print("\n" + "="*60)
    print("  GPU-Accelerated RAG — Evaluation Runner")
    print("="*60)

    documents, qa_pairs = get_data(force_reload=rebuild)
    qa_pairs            = qa_pairs[:n_questions]
    print(f"\n📋 Evaluating on {len(qa_pairs)} questions | {len(documents)} indexed docs")

    tokenizer, model = load_model()
    all_metrics = []

    if 1 in systems:
        print("\n" + "─"*60)
        print("🔹 Running System 1 — Vanilla LLM")
        results1 = run_vanilla(qa_pairs, tokenizer, model)
        save_results(results1, "system1_vanilla")
        all_metrics.append(compute_all_metrics(results1, "1_Vanilla_LLM"))

    if 2 in systems:
        print("\n" + "─"*60)
        print("🔹 Running System 2 — CPU RAG")
        results2 = run_cpu_rag(qa_pairs, documents, tokenizer, model)
        save_results(results2, "system2_cpu_rag")
        all_metrics.append(compute_all_metrics(results2, "2_CPU_RAG"))

    if 3 in systems:
        print("\n" + "─"*60)
        print("🔹 Running System 3 — GPU RAG")
        results3 = run_gpu_rag(qa_pairs, documents, tokenizer, model, use_batching=True)
        save_results(results3, "system3_gpu_rag")
        all_metrics.append(compute_all_metrics(results3, "3_GPU_RAG"))

    if all_metrics:
        print_metrics_table(all_metrics)
        save_metrics_csv(all_metrics)

    print("✅ Evaluation complete!")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems",     nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_questions", type=int,  default=500)
    parser.add_argument("--rebuild",     action="store_true")
    args = parser.parse_args()
    main(systems=args.systems, n_questions=args.n_questions, rebuild=args.rebuild)