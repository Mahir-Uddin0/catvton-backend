FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CATVTON_ROOT=/service/CatVTON \
    DEVICE=cuda \
    PORT=8080

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3.10 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY requirements.txt ./
RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install -r requirements.txt

# Only the Mask-Free pipeline's dependencies are included, not the masking stack.
COPY CatVTON/model/pipeline.py CatVTON/model/
COPY CatVTON/model/attn_processor.py CatVTON/model/
COPY CatVTON/model/utils.py CatVTON/model/
COPY CatVTON/utils.py CatVTON/
COPY app ./app

EXPOSE 8080

CMD ["python3.10", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
