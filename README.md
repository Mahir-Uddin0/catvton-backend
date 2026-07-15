# CatVTON Mask-Free API

This FastAPI service lives beside the CatVTON source checkout and loads only
CatVTONPix2PixPipeline. It does not load Detectron2, DensePose, SCHP, Gradio,
or the mask-based pipeline.

## Run locally

Create the virtual environment at this repository root:

    python3.9 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    uvicorn app.main:app --host 0.0.0.0 --port 8000

The first startup downloads the base model, VAE, and CatVTON Mask-Free
attention checkpoint. CPU is useful only as a smoke test.

## Endpoint

POST /v1/try-on accepts multipart form data:

- person_image: image file
- cloth_image: image file
- steps: optional integer from 1 to 100, default 50
- guidance_scale: optional float from 0 to 10, default 2.5
- seed: optional integer, default 42

It returns image/png. GET /healthz reports success once the model is loaded.

    curl --request POST http://localhost:8000/v1/try-on \
      --form person_image=@CatVTON/inputs/person.jpg \
      --form cloth_image=@CatVTON/inputs/cloth.jpg \
      --form steps=2 \
      --form guidance_scale=1.0 \
      --form seed=42 \
      --output tryon.png

## Docker / Cloud Run

    docker build --tag catvton-api .
    docker run --rm --gpus all --publish 8080:8080 \
      --env DEVICE=cuda \
      --env ALLOWED_ORIGINS=https://your-frontend.example \
      catvton-api

The image is CUDA-ready and starts one Uvicorn worker, so one container loads
one model copy. Deploy Cloud Run with one GPU and a configured CORS origin.
