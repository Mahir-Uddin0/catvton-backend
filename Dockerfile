FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CATVTON_ROOT=/service/CatVTON \
    DEVICE=auto \
    PORT=8000 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3.10 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY requirements.txt ./
RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install -r requirements.txt

# Pre-download model weights during build so the runtime does not need HF at startup
RUN python3.10 - <<'PY'
from huggingface_hub import snapshot_download
cache_dir = "/root/.cache/huggingface"
for repo in [
    "timbrooks/instruct-pix2pix",
    "zhengchong/CatVTON-MaskFree",
    "stabilityai/sd-vae-ft-mse",
]:
    snapshot_download(repo_id=repo, cache_dir=cache_dir, local_dir_use_symlinks=False)
PY

COPY CatVTON/model/pipeline.py CatVTON/model/
COPY CatVTON/model/attn_processor.py CatVTON/model/
COPY CatVTON/model/utils.py CatVTON/model/
COPY CatVTON/utils.py CatVTON/
COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "python3.10 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]