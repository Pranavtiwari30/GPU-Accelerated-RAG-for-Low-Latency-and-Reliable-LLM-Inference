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
| **Dataset** | TriviaQA `rc` — 5,000 documents, 500 QA evaluation pairs |
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
| **GPU vs CPU compute** | Embedding: 2ms (GPU) vs 120ms (CPU) — 60x measured speedup |
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
├── Dockerfile                       # NGC PyTorch base container
├── docker-compose.yml               # GPU-enabled compose with Jupyter profile
├── requirements.txt
├── README.md
│
├── data/
│   ├── load_dataset.py              # TriviaQA download + cache
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
│   └── run_eval.py                  # Master evaluation runner
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
pip install transformers accelerate sentence-transformers datasets \
            numpy pandas tqdm scikit-learn matplotlib seaborn psutil pynvml faiss-cpu
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

### Download and cache the dataset
```bash
python data/load_dataset.py
```
Downloads TriviaQA, extracts 5,000 document chunks and 500 QA pairs into `data/processed/`.

### Test a single system
```bash
python system1_vanilla/vanilla_llm.py
python system2_cpu_rag/cpu_rag_pipeline.py
python system3_gpu_rag/gpu_rag_pipeline.py
```

### Run full evaluation (500 questions, all 3 systems)
```bash
python evaluation/run_eval.py
```

### Quick test (20 questions)
```bash
python evaluation/run_eval.py --n_questions 20
```

### Run specific systems only
```bash
python evaluation/run_eval.py --systems 1 3    # skips CPU RAG
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
| Load | 21.5 | 106.9 | 0.20x |
| Clean | 2,332.4 | 2,143.6 | 1.09x |
| Filter | 4,965.7 | 5,904.8 | 0.84x |
| Chunk | 10,948.0 | 11,185.1 | 0.98x |
| Dedup | 222.1 | 1,405.7 | 0.16x |
| Stats | 6,011.4 | 10,077.1 | 0.60x |
| Export | 2,154.4 | 2,434.0 | 0.89x |
| **TOTAL** | **26,655 ms** | **33,257 ms** | **0.80x** |

**Finding:** Pandas outperformed the GPU pipeline on text-heavy ETL. This is expected — CPU-GPU data transfer overhead dominates when operations are string-based rather than numerical. The GPU advantage emerges for large-scale numerical array operations (embeddings, matrix ops), not NLP text preprocessing. This is a core insight from ADS: GPU acceleration requires workload-specific analysis, not blanket application.

---

## Results

All results from a 500-question evaluation on RTX 4050 6GB.

### Performance

| System | Avg Latency | Throughput | GPU Memory |
|--------|-------------|------------|------------|
| 1 — Vanilla LLM | 2,477 ms | 0.404 q/s | 3,423 MB |
| 2 — CPU RAG | 3,372 ms | 0.297 q/s | 3,902 MB |
| **3 — GPU RAG** | **1,763 ms** | **0.567 q/s** | 5,917 MB |

### Latency Breakdown (RAG Systems)

| Component | CPU RAG | GPU RAG | Speedup |
|-----------|---------|---------|---------|
| Embedding | 120.0 ms | 2.0 ms | **60x** |
| Retrieval (FAISS) | 0.5 ms | 0.3 ms | 1.7x |
| LLM Generation | 3,249 ms | 1,759 ms | 1.85x |
| **End-to-end** | **3,372 ms** | **1,763 ms** | **1.91x** |

### Reliability

| System | Hallucination Rate | Factual Consistency | Answer Grounding |
|--------|--------------------|---------------------|------------------|
| 1 — Vanilla LLM | 57.6% | 42.4% | N/A |
| 2 — CPU RAG | 91.8% | 8.2% | 65.9% |
| 3 — GPU RAG | 91.8% | 8.2% | 65.9% |

> Open `notebooks/results_dashboard.html` for interactive charts covering all metrics.

---

## Key Findings & Analysis

### 1. GPU embedding is the dominant speedup
Moving MiniLM from CPU to GPU delivers a **60x speedup** on embedding (120ms → 2ms). This is where GPU parallelism is most effective — dense matrix multiplications across 384-dimensional vectors map perfectly to CUDA cores. End-to-end, GPU RAG is **1.91x faster** than CPU RAG and **1.41x faster** than vanilla LLM.

### 2. Generation is the real bottleneck
LLM generation accounts for ~99% of total latency across all systems (1,759ms out of 1,763ms in GPU RAG). Embedding (2ms) and retrieval (0.3ms) are negligible. This is a roofline insight — the system is compute-bound at generation, not memory-bound at retrieval. Further optimization should target generation (quantization, speculative decoding) rather than retrieval.

### 3. Context distraction — a RAG failure mode
RAG hallucination (91.8%) is significantly higher than vanilla LLM (57.6%). This is the **context distraction** problem: TriviaQA questions require precise factual answers, but Wikipedia passages retrieved via semantic similarity contain topically related but non-answering content. The model conditions on irrelevant context and performs worse than with no context at all. The 65.9% answer grounding score confirms the answers exist in retrieved docs — the model simply fails to extract them. This is an active research problem in RAG literature.

### 4. GPU ETL is not always faster
The ETL benchmark produced a counterintuitive result — Pandas was 1.25x faster than the GPU pipeline for text preprocessing. Data transfer overhead between CPU and GPU dominates for string-heavy, moderate-sized workloads. Rapids/cuDF is designed for large numerical dataframes, not NLP string operations. This finding illustrates a key ADS principle: profiling before optimizing.

### 5. 6GB VRAM is sufficient but tight
Peak GPU memory was 5,917 MB out of 6,144 MB available (96.3% utilization). The system remained stable across 500 queries. Float16 precision was the key enabler — float32 would have exceeded VRAM limits.

---

## Docker Deployment

The project is containerized using NVIDIA NGC's PyTorch base image — the same infrastructure used in production GPU deployments.

### Build and run
```bash
# Build image
docker build -t gpu-rag .

# Run evaluation with persistent results
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  gpu-rag python evaluation/run_eval.py --n_questions 500

# Launch Jupyter for interactive exploration
docker-compose --profile jupyter up
# Open: http://localhost:8888
```

### NGC base image
```
FROM nvcr.io/nvidia/pytorch:24.01-py3
```

Using NGC ensures CUDA, cuDNN, and NCCL are pre-configured and version-matched — eliminating the environment setup issues encountered during local Windows development.

---

## Limitations & Future Work

| Limitation | Cause | Future Fix |
|---|---|---|
| No 4-bit quantization | bitsandbytes Windows bug | Run on Linux / WSL2 |
| FAISS runs on CPU | faiss-gpu requires Linux | Deploy via Docker on Linux |
| High RAG hallucination | Context distraction on TriviaQA | Add re-ranking / cross-encoder |
| Small model (1.1B) | 6GB VRAM constraint | Use Mistral-7B on cloud GPU |
| ETL GPU underperformance | cuDF 25.10 string kernel bug on Kaggle | Use stable Rapids environment |

**Potential extensions:**
- Replace FAISS with cuVS (NVIDIA's GPU-native vector search)
- Add cross-encoder re-ranking to fix context distraction
- Implement speculative decoding for faster generation
- Scale to 50,000+ documents using Dask distributed embedding
- Serve as a REST API inside Docker with NGINX reverse proxy

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Johnson et al. (2019). *Billion-scale similarity search with GPUs.* IEEE TPAMI. (FAISS paper)
- NVIDIA Rapids Documentation — https://rapids.ai
- TinyLlama Model — https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- TriviaQA Dataset — https://huggingface.co/datasets/trivia_qa
- NVIDIA NGC Containers — https://catalog.ngc.nvidia.com
