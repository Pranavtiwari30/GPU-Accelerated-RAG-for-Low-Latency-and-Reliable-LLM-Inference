"""
evaluation/run_eval.py

Master evaluation script. Runs all 3 systems and saves results.

Usage:
    python evaluation/run_eval.py                            # TriviaQA, 500 questions
    python evaluation/run_eval.py --dataset squad            # SQuAD, 500 questions
    python evaluation/run_eval.py --dataset squad --n_questions 50
    python evaluation/run_eval.py --systems 1 3
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
from system2_cpu_rag.cpu_rag_pipeline import (
    run_pipeline as run_cpu_rag,
    INDEX_PATH_TRIVIAQA as CPU_INDEX_TRIVIAQA,
    INDEX_PATH_SQUAD    as CPU_INDEX_SQUAD,
)
from system3_gpu_rag.gpu_rag_pipeline import (
    run_pipeline as run_gpu_rag,
    INDEX_PATH_TRIVIAQA as GPU_INDEX_TRIVIAQA,
    INDEX_PATH_SQUAD    as GPU_INDEX_SQUAD,
)
from evaluation.metrics import compute_all_metrics, print_metrics_table

RESULTS_DIR = "./results"


# ─── SQuAD loader ─────────────────────────────────────────────────────────────

def get_squad_data(n_questions: int = 500):
    """
    Load SQuAD v1.1 validation set.
    Returns (documents, qa_pairs) in the same format as get_data().

    Key advantage over TriviaQA:
      - Each passage DIRECTLY contains the answer as a text span
      - Eliminates context-distraction failure mode
      - RAG expected to show lower hallucination than Vanilla
    """
    from datasets import load_dataset

    print("📥 Loading SQuAD v1.1 validation set...")
    squad = (
        load_dataset("squad", split="validation")
        .shuffle(seed=42)
        .select(range(n_questions))
    )

    documents     = []
    qa_pairs      = []
    seen_contexts = {}

    for item in squad:
        ctx = item["context"]
        if ctx not in seen_contexts:
            seen_contexts[ctx] = len(documents)
            documents.append({
                "text":   ctx,
                "source": item["title"],
                "id":     len(documents),
            })

        qa_pairs.append({
            "question":       item["question"],
            "ground_truth":   item["answers"]["text"][0],
            "answer_aliases": item["answers"]["text"],
            "context_id":     seen_contexts[ctx],
        })

    print(f"✅ SQuAD loaded: {len(qa_pairs)} questions | {len(documents)} unique passages")
    print(f"   Example Q : {qa_pairs[0]['question']}")
    print(f"   Example GT: {qa_pairs[0]['ground_truth']}")
    return documents, qa_pairs


# ─── Save helpers ─────────────────────────────────────────────────────────────

def save_results(results: list, system_name: str, dataset_tag: str = ""):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag       = f"_{dataset_tag}" if dataset_tag else ""
    path      = os.path.join(RESULTS_DIR, f"{system_name}{tag}_{timestamp}.json")
    slim = [{k: v for k, v in r.items() if k != "retrieved_docs"} for r in results]
    with open(path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"💾 Results saved → {path}")
    return path


def save_metrics_csv(all_metrics: list, dataset_tag: str = ""):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag       = f"_{dataset_tag}" if dataset_tag else ""
    path      = os.path.join(RESULTS_DIR, f"metrics_comparison{tag}_{timestamp}.csv")
    pd.DataFrame(all_metrics).to_csv(path, index=False)
    print(f"💾 Metrics CSV saved → {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(systems=(1, 2, 3), n_questions=500, rebuild=False, dataset="triviaqa"):
    print("\n" + "=" * 60)
    print("  GPU-Accelerated RAG — Evaluation Runner")
    print(f"  Dataset : {dataset.upper()}")
    print(f"  Systems : {systems}")
    print(f"  N       : {n_questions}")
    print("=" * 60)

    # ── Select dataset + index paths ──────────────────────────────────────────
    if dataset == "squad":
        documents, qa_pairs = get_squad_data(n_questions=n_questions)
        dataset_tag         = "squad"
        cpu_index_path      = CPU_INDEX_SQUAD
        gpu_index_path      = GPU_INDEX_SQUAD
        # Always rebuild index for SQuAD — never reuse TriviaQA cached index
        force_rebuild       = True
    else:
        documents, qa_pairs = get_data(force_reload=rebuild)
        qa_pairs            = qa_pairs[:n_questions]
        dataset_tag         = "triviaqa"
        cpu_index_path      = CPU_INDEX_TRIVIAQA
        gpu_index_path      = GPU_INDEX_TRIVIAQA
        force_rebuild       = rebuild

    print(f"\n📋 Evaluating  : {len(qa_pairs)} questions")
    print(f"   Documents   : {len(documents)}")
    print(f"   CPU index   : {cpu_index_path}")
    print(f"   GPU index   : {gpu_index_path}")
    print(f"   Rebuild idx : {force_rebuild}")

    tokenizer, model = load_model()
    all_metrics      = []

    # ── System 1 — Vanilla LLM ────────────────────────────────────────────────
    if 1 in systems:
        print("\n" + "─" * 60)
        print("🔹 Running System 1 — Vanilla LLM")
        results1 = run_vanilla(qa_pairs, tokenizer, model)
        save_results(results1, "system1_vanilla", dataset_tag)
        all_metrics.append(compute_all_metrics(results1, "1_Vanilla_LLM"))

    # ── System 2 — CPU RAG ────────────────────────────────────────────────────
    if 2 in systems:
        print("\n" + "─" * 60)
        print("🔹 Running System 2 — CPU RAG")
        results2 = run_cpu_rag(
            qa_pairs, documents, tokenizer, model,
            index_path=cpu_index_path,
            force_rebuild=force_rebuild,
        )
        save_results(results2, "system2_cpu_rag", dataset_tag)
        all_metrics.append(compute_all_metrics(results2, "2_CPU_RAG"))

    # ── System 3 — GPU RAG ────────────────────────────────────────────────────
    if 3 in systems:
        print("\n" + "─" * 60)
        print("🔹 Running System 3 — GPU RAG")
        results3 = run_gpu_rag(
            qa_pairs, documents, tokenizer, model,
            use_batching=True,
            index_path=gpu_index_path,
            force_rebuild=force_rebuild,
        )
        save_results(results3, "system3_gpu_rag", dataset_tag)
        all_metrics.append(compute_all_metrics(results3, "3_GPU_RAG"))

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_metrics:
        print_metrics_table(all_metrics)
        save_metrics_csv(all_metrics, dataset_tag)

    print(f"\n✅ Evaluation complete! [{dataset.upper()} · {len(qa_pairs)} questions]")
    return all_metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems",     nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_questions", type=int,            default=500)
    parser.add_argument("--rebuild",     action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="triviaqa",
        choices=["triviaqa", "squad"],
        help="'triviaqa' (original) or 'squad' (answer-in-context, fixes hallucination)",
    )
    args = parser.parse_args()
    main(
        systems=args.systems,
        n_questions=args.n_questions,
        rebuild=args.rebuild,
        dataset=args.dataset,
    )