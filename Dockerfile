# ── GPU-Accelerated RAG Project ───────────────────────────────────────────────
#
# TWO BUILD TARGETS:
#
#   Production (NGC base, full GPU support):
#     docker build --target production -t gpu-rag .
#     docker run --gpus all -v $(pwd)/results:/app/results gpu-rag
#
#   CI / CPU-only (lightweight, no GPU required):
#     docker build --target ci -t gpu-rag-ci .
#     docker run gpu-rag-ci
#
# NGC base image connects to NVIDIA's container registry used in
# production GPU deployments (covered in ADS course).
# ─────────────────────────────────────────────────────────────────────────────

ARG CI_BUILD=false

# ── Stage 1: CI base (lightweight, used by GitHub Actions) ───────────────────
FROM python:3.10-slim AS ci

LABEL maintainer="ADS Project — GPU RAG"
LABEL description="GPU-Accelerated RAG — CI build (CPU only)"

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY setup/requirements.txt .

# CPU-only deps for CI validation
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    --no-deps bitsandbytes || true && \
    pip install --no-cache-dir \
    transformers>=4.37.0 \
    accelerate>=0.26.0 \
    sentence-transformers>=2.3.0 \
    faiss-cpu>=1.7.4 \
    datasets>=2.16.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    tqdm>=4.66.0 \
    scikit-learn>=1.3.0 \
    psutil>=5.9.0

COPY data/            ./data/
COPY system1_vanilla/ ./system1_vanilla/
COPY system2_cpu_rag/ ./system2_cpu_rag/
COPY system3_gpu_rag/ ./system3_gpu_rag/
COPY evaluation/      ./evaluation/
COPY notebooks/       ./notebooks/

RUN mkdir -p /app/results

# Validate imports only — no GPU needed
CMD ["python", "-c", "\
import torch; \
from transformers import AutoTokenizer; \
from sentence_transformers import SentenceTransformer; \
import faiss, datasets, pandas, numpy; \
print('✅ All imports OK'); \
print(f'   PyTorch: {torch.__version__}'); \
print(f'   CUDA available: {torch.cuda.is_available()}') \
"]

# ── Stage 2: Production (NGC PyTorch, full GPU) ───────────────────────────────
FROM nvcr.io/nvidia/pytorch:24.01-py3 AS production

LABEL maintainer="ADS Project — GPU RAG"
LABEL description="GPU-Accelerated RAG — Production build (NGC + CUDA)"

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY setup/requirements.txt .

RUN pip install --no-cache-dir \
    transformers>=4.37.0 \
    accelerate>=0.26.0 \
    sentence-transformers>=2.3.0 \
    faiss-cpu>=1.7.4 \
    datasets>=2.16.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    tqdm>=4.66.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.13.0 \
    psutil>=5.9.0 \
    pynvml>=11.5.0 \
    jupyter>=1.0.0

COPY data/            ./data/
COPY system1_vanilla/ ./system1_vanilla/
COPY system2_cpu_rag/ ./system2_cpu_rag/
COPY system3_gpu_rag/ ./system3_gpu_rag/
COPY evaluation/      ./evaluation/
COPY notebooks/       ./notebooks/

RUN mkdir -p /app/results/charts

EXPOSE 8888

CMD ["python", "evaluation/run_eval.py", "--n_questions", "500"]