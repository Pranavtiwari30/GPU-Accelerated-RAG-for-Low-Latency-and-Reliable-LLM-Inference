"""
data/load_dataset.py

Loads TriviaQA (rc subset) from HuggingFace.
Produces:
  - documents : List[str]  → used to build FAISS index
  - qa_pairs  : List[dict] → used for evaluation
                 [{"question": ..., "answer": ..., "answer_aliases": [...]}, ...]

Run standalone to verify and cache the dataset:
  python data/load_dataset.py
"""

import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR   = "./data/cache"
OUTPUT_DIR  = "./data/processed"
MAX_DOCS    = 5000   # documents fed into FAISS index
MAX_QA      = 500    # QA pairs used for evaluation (keep small for speed)
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

def load_triviaqa(max_docs=MAX_DOCS, max_qa=MAX_QA, seed=RANDOM_SEED):
    """
    Returns (documents, qa_pairs).

    documents  : list of plain-text strings (paragraphs from TriviaQA sources)
    qa_pairs   : list of dicts with keys: question, answer, answer_aliases
    """
    print("📥 Loading TriviaQA (rc subset) from HuggingFace...")
    # rc = reading comprehension split — has source documents attached
    dataset = load_dataset("trivia_qa", "rc", cache_dir=CACHE_DIR, trust_remote_code=True)

    train_data = dataset["train"]
    random.seed(seed)
    indices = random.sample(range(len(train_data)), min(max_qa * 4, len(train_data)))
    subset  = train_data.select(indices)

    documents = []
    qa_pairs  = []

    print("🔨 Extracting documents and QA pairs...")
    seen_docs = set()

    for item in tqdm(subset):
        # ── Extract QA pair ───────────────────────────────────────────────────
        question = item["question"]
        answer   = item["answer"]["value"]                        # primary answer
        aliases  = item["answer"].get("aliases", [])              # alternative correct answers

        if not question or not answer:
            continue

        # ── Extract source documents ──────────────────────────────────────────
        # TriviaQA rc provides search results as source passages
        search_results = item.get("search_results", {})
        passages = search_results.get("search_context", [])

        for passage in passages:
            if isinstance(passage, str) and len(passage) > 100:
                # Deduplicate and chunk into ~300 word paragraphs
                chunks = chunk_text(passage, chunk_size=300)
                for chunk in chunks:
                    if chunk not in seen_docs:
                        seen_docs.add(chunk)
                        documents.append(chunk)

        if len(qa_pairs) < max_qa:
            qa_pairs.append({
                "question":       question,
                "answer":         answer,
                "answer_aliases": aliases
            })

        if len(documents) >= max_docs and len(qa_pairs) >= max_qa:
            break

    # Trim to limits
    documents = documents[:max_docs]
    qa_pairs  = qa_pairs[:max_qa]

    print(f"✅ Loaded {len(documents)} documents and {len(qa_pairs)} QA pairs")
    return documents, qa_pairs


def chunk_text(text, chunk_size=300):
    """Split text into ~chunk_size word chunks with slight overlap."""
    words  = text.split()
    chunks = []
    step   = int(chunk_size * 0.9)   # 10% overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) > 50:   # skip tiny trailing chunks
            chunks.append(chunk)
    return chunks


def save_processed(documents, qa_pairs, output_dir=OUTPUT_DIR):
    """Save processed data to disk so you don't reload every run."""
    os.makedirs(output_dir, exist_ok=True)

    docs_path = os.path.join(output_dir, "documents.json")
    qa_path   = os.path.join(output_dir, "qa_pairs.json")

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved documents -> {docs_path}")
    print(f"Saved QA pairs  -> {qa_path}")


def load_processed(output_dir=OUTPUT_DIR):
    """Load previously saved processed data (fast reload)."""
    docs_path = os.path.join(output_dir, "documents.json")
    qa_path   = os.path.join(output_dir, "qa_pairs.json")

    if not os.path.exists(docs_path) or not os.path.exists(qa_path):
        raise FileNotFoundError(
            "Processed data not found. Run load_dataset.py first."
        )

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(documents)} documents and {len(qa_pairs)} QA pairs from cache")
    return documents, qa_pairs


def get_data(force_reload=False):
    """
    Main entry point used by all 3 pipeline systems.
    Returns (documents, qa_pairs).
    Uses cached version if available.
    """
    docs_path = os.path.join(OUTPUT_DIR, "documents.json")

    if not force_reload and os.path.exists(docs_path):
        return load_processed()
    else:
        documents, qa_pairs = load_triviaqa()
        save_processed(documents, qa_pairs)
        return documents, qa_pairs


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs, qas = get_data(force_reload=False)

    print("\n── Sample Document ──────────────────────────────────────────────────")
    print(docs[0][:400])

    print("\n── Sample QA Pairs ──────────────────────────────────────────────────")
    for qa in qas[:3]:
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}")
        print(f"  Aliases: {qa['answer_aliases'][:3]}")
        print()
