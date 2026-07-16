FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CATVTON_ROOT=/service/CatVTON \
    DEVICE=auto \
    PORT=8000 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

# Install system dependencies
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3.10 \
        python3-pip \
        ca-certificates \
        git \
        git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /service

# Install Python dependencies
COPY requirements.txt ./
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r requirements.txt

# Download Hugging Face models during image build. Keep this online so the model cache is populated at build time.
COPY scripts/download_models.py /tmp/download_models.py
RUN python3.10 /tmp/download_models.py

# Copy application source
COPY CatVTON/model/pipeline.py CatVTON/model/
COPY CatVTON/model/attn_processor.py CatVTON/model/
COPY CatVTON/model/utils.py CatVTON/model/
COPY CatVTON/utils.py CatVTON/
COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "python3.10 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]