# GPU-Accelerated RAG for Low-Latency and Reliable LLM Inference

![Docker Build](https://github.com/Pranavtiwari30/GPU-Accelerated-RAG-for-Low-Latency-and-Reliable-LLM-Inference/actions/workflows/docker-build.yml/badge.svg)

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [ADS Concepts Applied](#ads-concepts-applied)
- [Hardware & Environment](#hardware--environment)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [ETL Pipeline — Rapids Benchmark](#etl-pipeline--rapids-benchmark)
- [Results](#results)
- [Key Findings & Analysis](#key-findings--analysis)
- [Docker Deployment](#docker-deployment)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Overview

This project builds and benchmarks three LLM inference pipelines to evaluate the impact of GPU acceleration on latency, throughput, and reliability in a Retrieval-Augmented Generation (RAG) system.

**Core question:** Does moving embedding, retrieval, and inference to GPU meaningfully reduce latency and improve reliability compared to a CPU-only RAG and a vanilla LLM baseline?

| | |
|---|---|
| **Model** | TinyLlama-1.1B-Chat-v1.0 (float16, ~2.2GB VRAM) |
| **Embedding** | all-MiniLM-L6-v2 (384-dim) |
| **Dataset** | SQuAD v1.1 — 500 QA evaluation pairs |
| **Local GPU** | NVIDIA RTX 4050 6GB |
| **ETL GPU** | Kaggle P100 16GB |

---

## System Architecture

Three pipelines were implemented and benchmarked end-to-end:

```
System 1 — Vanilla LLM (Baseline)
  Query ──► Tokenize ──► TinyLlama ──► Output

System 2 — CPU RAG
  Query ──► MiniLM (CPU) ──► FAISS CPU ──► Context + Query ──► TinyLlama ──► Output

System 3 — GPU RAG
  Query ──► MiniLM (CUDA) ──► FAISS ──► Context + Query ──► TinyLlama (batched) ──► Output
```

System 3 differs from System 2 in three ways:
- Embedding runs entirely on CUDA instead of CPU
- Queries are batched (4 at a time) for parallel GPU inference
- All tensor operations stay on GPU throughout — no CPU-GPU transfers mid-pipeline

---

## ADS Concepts Applied

| Course Topic | How It Was Applied |
|---|---|
| **GPU vs CPU compute** | Embedding: 2ms (GPU) vs 70ms (CPU) — 35x measured speedup |
| **Batched inference** | System 3 processes 4 queries simultaneously using CUDA parallelism |
| **Float16 / Mixed precision** | TinyLlama loaded in float16 — halves VRAM, uses Tensor Cores |
| **CuPy / Rapids** | ETL pipeline benchmarked with CuPy GPU arrays vs Pandas on Kaggle P100 |
| **FAISS vector search** | Hardware-accelerated similarity search for document retrieval |
| **Memory-bound vs compute-bound** | Roofline analysis — generation dominates, not retrieval |
| **Docker / NGC containers** | Dockerfile built on `nvcr.io/nvidia/pytorch` NGC base image |
| **ETL pipeline design** | 7-stage document preprocessing pipeline benchmarked at 191k chunks |

---

## Hardware & Environment

| Component | Spec |
|---|---|
| CPU | Intel Core i7 13th Gen |
| GPU (local) | NVIDIA RTX 4050 — 6GB VRAM |
| RAM | 16GB DDR5 |
| OS | Windows 11 |
| CUDA | 12.1 |
| Python | 3.10 |
| PyTorch | 2.1+ |
| ETL Benchmark | Kaggle P100 — 16GB VRAM |

---

## Project Structure

```
gpu_rag_project/
│
├── Dockerfile                       # Two-stage: CI (python:3.10-slim) + Production (NGC PyTorch)
├── docker-compose.yml               # GPU-enabled compose with Jupyter profile
├── README.md
│
├── setup/
│   └── requirements.txt             # Project dependencies
│
├── data/
│   ├── load_dataset.py              # Dataset download + cache
│   └── etl_pipeline.py              # Pandas vs CuPy ETL benchmark
│
├── system1_vanilla/
│   └── vanilla_llm.py               # Baseline: TinyLlama, no retrieval
│
├── system2_cpu_rag/
│   ├── embedder.py                  # MiniLM on CPU
│   ├── faiss_cpu.py                 # FAISS flat IP index
│   └── cpu_rag_pipeline.py          # Full CPU RAG pipeline
│
├── system3_gpu_rag/
│   ├── embedder_gpu.py              # MiniLM on CUDA
│   ├── faiss_gpu.py                 # FAISS GPU (CPU fallback on Windows)
│   └── gpu_rag_pipeline.py          # Batched GPU RAG pipeline
│
├── evaluation/
│   ├── metrics.py                   # Performance + reliability metrics
│   └── run_eval.py                  # Master evaluation runner (TriviaQA + SQuAD)
│
├── notebooks/
│   └── results_dashboard.html       # Interactive results (open in any browser)
│
└── results/
    ├── metrics_comparison_*.csv
    ├── system1_vanilla_*.json
    ├── system2_cpu_rag_*.json
    └── system3_gpu_rag_*.json
```

---

## Setup & Installation

### 1. Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install all other dependencies
```bash
pip install -r setup/requirements.txt
```

### 3. Verify GPU is detected
```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4050
```

> **Note — bitsandbytes on Windows:** 4-bit quantization has a known Windows bug where weights
> materialize on GPU before quantization, causing OOM on 6GB cards. This project uses TinyLlama
> in float16 instead — stable on Windows, same inference pipeline.

> **Note — FAISS GPU on Windows:** `faiss-gpu` requires Linux. The codebase auto-detects and
> falls back to CPU FAISS silently. GPU FAISS works on Kaggle/Colab (Linux).

---

## Usage

### Run full evaluation — SQuAD v1.1 (default, 500 questions)
```bash
python evaluation/run_eval.py --dataset squad
```

### Run full evaluation — TriviaQA (original benchmark)
```bash
python evaluation/run_eval.py --dataset triviaqa
```

### Quick test (50 questions)
```bash
python evaluation/run_eval.py --dataset squad --n_questions 50
```

### Run specific systems only
```bash
python evaluation/run_eval.py --dataset squad --systems 1 3    # skips CPU RAG
```

### Test a single system
```bash
python system1_vanilla/vanilla_llm.py
python system2_cpu_rag/cpu_rag_pipeline.py
python system3_gpu_rag/gpu_rag_pipeline.py
```

### View interactive results
Open `notebooks/results_dashboard.html` in any browser — no server or Python needed.

---

## ETL Pipeline — Rapids Benchmark

A 7-stage document ETL pipeline was built in `data/etl_pipeline.py` and benchmarked comparing **Pandas (CPU)** against **CuPy GPU arrays** on a Kaggle P100 across 191,224 document chunks.

Kaggle ETL notebook → [View on Kaggle](https://www.kaggle.com/code/pranavtiwari30102003/etl-pipleline)

### ETL Results — Kaggle P100, 191k chunks

| Operation | Pandas CPU (ms) | CuPy GPU (ms) | Speedup |
|-----------|----------------|---------------|---------|
| Load      | 21.5           | 106.9         | 0.20x   |
| Clean     | 2,332.4        | 2,143.6       | 1.09x   |
| Filter    | 4,965.7        | 5,904.8       | 0.84x   |
| Chunk     | 10,948.0       | 11,185.1      | 0.98x   |
| Dedup     | 222.1          | 1,405.7       | 0.16x   |
| Stats     | 6,011.4        | 10,077.1      | 0.60x   |
| Export    | 2,154.4        | 2,434.0       | 0.89x   |
| **TOTAL** | **26,655 ms**  | **33,257 ms** | **0.80x** |

**Finding:** Pandas outperformed the GPU pipeline on text-heavy ETL. CPU-GPU data transfer overhead dominates when operations are string-based rather than numerical. The GPU advantage emerges for large-scale numerical array operations (embeddings, matrix ops), not NLP text preprocessing. This is a core ADS insight: GPU acceleration requires workload-specific analysis, not blanket application.

---

## Results

All results from a 500-question evaluation on RTX 4050 6GB using **SQuAD v1.1**.

### Performance

| System          | Avg Latency | Throughput  | GPU Memory |
|-----------------|-------------|-------------|------------|
| 1 — Vanilla LLM | 2,753 ms    | 0.363 q/s   | 3,429 MB   |
| 2 — CPU RAG     | 2,252 ms    | 0.444 q/s   | 5,681 MB   |
| **3 — GPU RAG** | **930 ms**  | **1.076 q/s** | 6,097 MB |

### Latency Breakdown (RAG Systems)

| Component         | CPU RAG   | GPU RAG  | Speedup   |
|-------------------|-----------|----------|-----------|
| Embedding         | 70.6 ms   | 2.9 ms   | **24x**   |
| Retrieval (FAISS) | 0.1 ms    | 0.4 ms   | —         |
| LLM Generation    | 2,243.7 ms | 805.3 ms | **2.79x** |
| **End-to-end**    | **2,252 ms** | **930 ms** | **2.42x** |

### Reliability — SQuAD v1.1

| System          | Hallucination Rate | Factual Consistency | Answer Grounding |
|-----------------|--------------------|---------------------|------------------|
| 1 — Vanilla LLM | 89.2%              | 10.8%               | N/A              |
| 2 — CPU RAG     | 58.0%              | 42.0%               | 99.5%            |
| **3 — GPU RAG** | **57.2%**          | **42.8%**           | **100%**         |

> RAG reduces hallucination by **32%** vs Vanilla LLM on SQuAD. Answer grounding at 100%
> confirms the retrieval pipeline correctly surfaces answer-containing passages.

> Open `notebooks/results_dashboard.html` for interactive charts covering all metrics.

---

## Key Findings & Analysis

### 1. GPU RAG is 2.96× faster than Vanilla LLM
GPU RAG (930ms) vs Vanilla LLM (2,753ms). Moving embedding and inference to GPU delivers substantial end-to-end speedup. GPU RAG is also **2.42× faster** than CPU RAG due to GPU embedding + batched inference.

### 2. GPU embedding delivers a 24× speedup
Moving MiniLM from CPU to GPU reduces embedding latency from 70.6ms to 2.9ms. Dense matrix multiplications across 384-dimensional vectors map perfectly to CUDA cores — this is where GPU parallelism is most effective.

### 3. RAG reduces hallucination by 32% on SQuAD
On SQuAD v1.1, where retrieved passages directly contain the answer span, RAG reduces hallucination from 89.2% (Vanilla) to 57.2% (GPU RAG). Answer grounding at 100% confirms the retrieval pipeline is working correctly — the model successfully extracts answers from retrieved context.

### 4. Generation is the real bottleneck
LLM generation accounts for ~99% of total latency (805ms out of 930ms in GPU RAG). Embedding (2.9ms) and retrieval (0.4ms) are negligible. This is a roofline insight — the system is compute-bound at generation, not retrieval. Further optimization should target generation (quantization, speculative decoding) rather than retrieval.

### 5. Context distraction on TriviaQA — a documented failure mode
Initial evaluation on TriviaQA revealed that RAG hallucination (91.8%) was higher than Vanilla (57.6%). This is the **context distraction** problem: TriviaQA requires precise factual answers, but Wikipedia passages retrieved via semantic similarity are topically related rather than directly answering. The model conditions on irrelevant context and performs worse than with no context at all. Switching to SQuAD — where passages contain the answer — confirmed the retrieval pipeline is correct. Context distraction is an active research problem in RAG literature; the fix is cross-encoder re-ranking or a larger model.

### 6. GPU ETL is not always faster
Pandas was 1.25× faster than the CuPy GPU pipeline for text preprocessing. Data transfer overhead dominates for string-heavy, moderate-sized workloads. Rapids/cuDF is designed for large numerical dataframes, not NLP string operations. Key ADS principle: profile before optimising.

### 7. 6GB VRAM is sufficient but tight
Peak GPU memory was 6,097 MB out of 6,144 MB available (99.2% utilisation). The system remained stable across 500 queries. Float16 precision was the key enabler — float32 would have exceeded VRAM limits.

---

## Docker Deployment

The project uses a **two-stage Dockerfile**:
- **CI stage** (`python:3.10-slim`) — lightweight, CPU-only, used by GitHub Actions
- **Production stage** (`nvcr.io/nvidia/pytorch:24.01-py3`) — full CUDA, NGC base image

### Build and run
```bash
# Production build (requires NVIDIA GPU)
docker build --target production -t gpu-rag .

# Run evaluation with persistent results
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  gpu-rag python evaluation/run_eval.py --dataset squad --n_questions 500

# CI build (CPU only, validates imports)
docker build --target ci -t gpu-rag-ci .
docker run gpu-rag-ci

# Launch Jupyter for interactive exploration
docker-compose --profile jupyter up
# Open: http://localhost:8888
```

### NGC base image
```
FROM nvcr.io/nvidia/pytorch:24.01-py3
```

Using NGC ensures CUDA, cuDNN, and NCCL are pre-configured and version-matched — eliminating environment setup issues encountered during local Windows development.

---

## Limitations & Future Work

| Limitation | Cause | Future Fix |
|---|---|---|
| No 4-bit quantization | bitsandbytes Windows bug | Run on Linux / WSL2 |
| FAISS runs on CPU | faiss-gpu requires Linux | Deploy via Docker on Linux |
| Small model (1.1B) | 6GB VRAM constraint | Use Mistral-7B on cloud GPU |
| Context distraction on TriviaQA | Model too small to extract from noisy context | Cross-encoder re-ranking |
| ETL GPU underperformance | String-heavy ops, transfer overhead | Use cuDF for numerical ETL |

**Potential extensions:**
- Replace FAISS with cuVS (NVIDIA's GPU-native vector search)
- Add cross-encoder re-ranking to resolve context distraction on TriviaQA
- Implement speculative decoding for faster generation
- Scale to 50,000+ documents using Dask distributed embedding
- Serve as a REST API inside Docker with NGINX reverse proxy

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Johnson et al. (2019). *Billion-scale similarity search with GPUs.* IEEE TPAMI. (FAISS paper)
- Rajpurkar et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP.
- NVIDIA Rapids Documentation — https://rapids.ai
- TinyLlama Model — https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- SQuAD Dataset — https://huggingface.co/datasets/squad
- NVIDIA NGC Containers — https://catalog.ngc.nvidia.com
