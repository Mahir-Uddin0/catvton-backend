# CatVTON Mask-Free Try-On API

A production-oriented FastAPI wrapper for the **CatVTON Mask-Free** virtual try-on model. Send a person image and a garment image, and receive a PNG of the person wearing the garment—without supplying a segmentation mask.

The service deliberately includes only the mask-free inference path. It does not load the original project's Gradio UI, Detectron2, DensePose, SCHP, or mask-based pipeline.


## Features

- Mask-free virtual try-on using `CatVTONPix2PixPipeline`
- FastAPI endpoint accepting standard multipart image uploads
- Deterministic generation when the same inputs and `seed` are used
- Configurable inference steps, guidance scale, resolution, device, and precision
- GPU-first CUDA 12.1 Docker image, with CPU available for functional smoke tests
- Automatic Hugging Face model download and configurable local cache
- Image type, image-decode, and upload-size validation
- Readiness endpoint and configurable browser CORS origins
- One model instance and serialized inference per process to avoid concurrent GPU contention

## Project structure

```text
.
├── app/
│   ├── main.py                 # FastAPI app, request validation, model lifecycle
│   └── config.py               # Environment-based settings
├── CatVTON/
│   ├── model/
│   │   ├── pipeline.py         # Mask-free diffusion pipeline
│   │   └── attn_processor.py   # CatVTON attention implementation
│   ├── utils.py                # Image resize, padding, and latent utilities
│   └── LICENSE                 # Upstream CatVTON license
├── scripts/
│   └── download_models.py      # Optional model-cache warm-up script
├── .env.example                # Local configuration template
├── Dockerfile                  # Single-worker CUDA runtime image
└── requirements.txt            # API and model runtime dependencies
```

## Architecture

```text
Client
  │  multipart/form-data: person image + garment image
  ▼
FastAPI  ── validation ──>  RGB decode + max-size check
  │
  │  one request at a time per process
  ▼
CatVTONPix2PixPipeline
  ├─ resize/crop person image to OUTPUT_WIDTH × OUTPUT_HEIGHT
  ├─ resize/pad garment image to the same canvas
  ├─ VAE encodes both images into latents
  ├─ CatVTON attention + DDIM diffusion denoising
  └─ VAE decodes the generated try-on image
  │
  ▼
PNG response + X-Inference-Seed header
```

At application startup, the service loads the base InstructPix2Pix components, Stable Diffusion VAE, and CatVTON Mask-Free attention checkpoint into one process-local pipeline. On a cache miss, Hugging Face downloads these assets into `HF_HOME`.

## Tech stack

| Area | Technology |
| --- | --- |
| HTTP API | FastAPI, Uvicorn, `python-multipart` |
| ML runtime | PyTorch 2.4, Accelerate |
| Diffusion | Diffusers, DDIM scheduler, Stable Diffusion VAE |
| Model | CatVTON Mask-Free / InstructPix2Pix attention checkpoint |
| Images | Pillow, NumPy |
| Packaging | Docker, NVIDIA CUDA 12.1 + cuDNN 8 runtime |

## API

`POST /v1/try-on`

Send `multipart/form-data` with the following fields:

| Field | Type | Required | Default | Constraints |
| --- | --- | --- | --- | --- |
| `person_image` | image file | Yes | — | Decodable image; at most `MAX_UPLOAD_BYTES` |
| `cloth_image` | image file | Yes | — | Decodable image; at most `MAX_UPLOAD_BYTES` |
| `steps` | integer | No | `50` | `1`–`100` |
| `guidance_scale` | number | No | `2.5` | `0`–`10` |
| `seed` | integer | No | `42` | `0`–`2147483647` |

A successful request returns `200 OK` with `Content-Type: image/png`. The seed used is also returned in the `X-Inference-Seed` response header.

```bash
curl --request POST http://localhost:8000/v1/try-on \
  --form person_image=@CatVTON/inputs/person.jpg \
  --form cloth_image=@CatVTON/inputs/cloth.jpg \
  --form steps=30 \
  --form guidance_scale=2.5 \
  --form seed=42 \
  --output try-on.png
```

Common error responses are `413` for oversized uploads, `415` for a non-image content type, `422` for an invalid image or invalid form parameters, `500` for inference failures, and `503` when the model is not ready.

### Health check

`GET /healthz` returns `{"status":"ok"}` only after the model is loaded; otherwise it returns `503`. Use it for container readiness checks.

## Run locally

### Prerequisites

- Python 3.10 recommended (the container uses Python 3.10)
- An NVIDIA GPU and compatible driver for practical inference
- CUDA-compatible PyTorch installation; `requirements.txt` installs the CUDA 12.1 PyTorch wheel
- Git LFS if you are cloning model assets or upstream files that require it

CPU execution is supported by the configuration, but image generation will be slow and is best treated as a smoke test.

### Setup

```bash
git clone https://github.com/Mahir-Uddin0/catvton-backend
cd catvton

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The initial startup downloads the required Hugging Face models. To download them before starting the server—for example while building a reusable machine image—run:

```bash
python scripts/download_models.py
```

Then verify the server once its startup logs finish:

```bash
curl http://localhost:8000/healthz
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Environment variables

Copy `.env.example` to `.env` for local development. Environment variables provided by the shell or deployment platform take precedence over values in `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. `auto` selects CUDA when available. |
| `MIXED_PRECISION` | `bf16` | `bf16`, `fp16`, or `no`. CPU always uses `float32`; CUDA falls back to `fp16` if BF16 is unavailable. |
| `OUTPUT_WIDTH` | `768` | Generated image width; must be a positive multiple of 8. |
| `OUTPUT_HEIGHT` | `1024` | Generated image height; must be a positive multiple of 8. |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum size, in bytes, accepted for each uploaded image. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins allowed to call the API from a browser. |
| `HF_HOME` | `~/.cache/huggingface` | Directory for downloaded model weights and Hugging Face cache. Mount this as persistent storage in containers. |
| `CATVTON_ROOT` | `./CatVTON` | Location of the bundled CatVTON source; it must contain `model/pipeline.py`. |
| `BASE_MODEL_ID` | `timbrooks/instruct-pix2pix` | Hugging Face base model identifier or compatible local path. |
| `ATTENTION_CHECKPOINT_ID` | `zhengchong/CatVTON-MaskFree` | Hugging Face CatVTON attention checkpoint identifier or compatible local path. |
| `PORT` | `8000` in Docker | Uvicorn port used by the Docker command. |

Example production-oriented configuration:

```dotenv
DEVICE=cuda
MIXED_PRECISION=bf16
ALLOWED_ORIGINS=https://app.example.com
OUTPUT_WIDTH=768
OUTPUT_HEIGHT=1024
MAX_UPLOAD_BYTES=10485760
HF_HOME=/cache/huggingface
```

## Deployment options

### Docker on a GPU host

Build the supplied CUDA image and expose the same port the container listens on:

```bash
docker build --tag catvton-api .
docker run --rm --gpus all \
  --publish 8080:8080 \
  --env PORT=8080 \
  --env DEVICE=cuda \
  --env ALLOWED_ORIGINS=https://app.example.com \
  --volume catvton-hf-cache:/cache/huggingface \
  catvton-api
```

The image starts **one** Uvicorn worker. Keep one worker per GPU/model replica; adding workers loads another full model copy and increases GPU memory use.


## Design decisions

- **Mask-free path only:** the API uses `CatVTONPix2PixPipeline`, avoiding segmentation/mask input and excluding heavyweight Detectron2, DensePose, SCHP, and Gradio dependencies from the serving runtime.
- **Load once at startup:** model construction is expensive, so the FastAPI lifespan hook creates one reusable pipeline and releases CUDA cache at shutdown.
- **Serialized inference:** an `asyncio.Lock` permits only one active generation per process. This keeps GPU memory predictable and avoids overlapping model calls; scale horizontally for concurrent requests.
- **Fixed output canvas:** person images are resized/cropped while garment images are resized/padded to the configured output dimensions. This matches pipeline requirements and makes memory demand predictable.
- **Seeded generator:** the API creates a per-request PyTorch generator, enabling reproducible runs where the underlying runtime and inputs are unchanged.
- **PNG response in memory:** inputs and outputs are not written by the API to an application directory. The only persistent artifacts are model-cache files in `HF_HOME` if that location is mounted.
- **GPU-first precision:** BF16 is preferred when supported; FP16 is used as a fallback, while CPU uses FP32 for compatibility.

## Security and production hardening

Built-in protections are intentionally narrow: the service validates the advertised image MIME type, attempts to decode every upload with Pillow, limits each upload size, and restricts browser origins through `ALLOWED_ORIGINS`.

Before exposing the API publicly, add the following:

- Authentication and authorization at an API gateway or reverse proxy; this application has no auth.
- Rate limits, concurrency quotas, request timeouts, and body-size limits upstream. GPU inference is expensive and the in-process lock makes it a natural denial-of-service target without them.
- TLS termination and a restrictive `ALLOWED_ORIGINS` value—never use a permissive origin unnecessarily.
- Malware/content scanning and image pixel-dimension limits. Byte limits alone do not prevent decompression-bomb-style images.
- Structured request logs and monitoring that avoid storing source images or other sensitive user data.
- Private model-cache storage and pinned model revisions/checksums for repeatable, supply-chain-aware deployments.

The current pipeline is created with `skip_safety_check=True`; it does **not** run the upstream safety checker. If content moderation is required, implement it explicitly before returning outputs and ensure the policy is appropriate for your users and jurisdiction.

## Future improvements

- **Quantization:** evaluate weight-only 8-bit/4-bit or FP8 inference where model quality and operator compatibility remain acceptable; benchmark quality, latency, and VRAM before adopting it.
- **Batch inference:** introduce a bounded queue and dynamic micro-batching for compatible resolution/step settings. The current service intentionally processes one request at a time.
- **Performance:** evaluate `torch.compile`, attention optimizations, TensorRT/ONNX paths where supported, and pre-warmed model caches.
- **Scalability:** add GPU-aware autoscaling, queue-depth metrics, request tracing, and separate API/worker processes.

