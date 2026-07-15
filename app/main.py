import asyncio
import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from app.config import load_settings

logger = logging.getLogger(__name__)
settings = load_settings()


def load_pipeline(app: FastAPI) -> None:
    root = settings.catvton_root
    if not (root / "model" / "pipeline.py").is_file():
        raise RuntimeError(f"CATVTON_ROOT is invalid: {root}")
    sys.path.insert(0, str(root))
    from model.pipeline import CatVTONPix2PixPipeline

    app.state.pipeline = CatVTONPix2PixPipeline(
        base_ckpt=os.getenv("BASE_MODEL_ID", "timbrooks/instruct-pix2pix"),
        attn_ckpt=os.getenv("ATTENTION_CHECKPOINT_ID", "zhengchong/CatVTON-MaskFree"),
        attn_ckpt_version="mix-48k-1024",
        weight_dtype=settings.dtype,
        device=settings.device,
        skip_safety_check=True,
        use_tf32=settings.device == "cuda" if torch.cuda.is_available() else "cpu",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading CatVTON Mask-Free on %s", settings.device)
    await run_in_threadpool(load_pipeline, app)
    app.state.lock, app.state.ready = asyncio.Lock(), True
    try:
        yield
    finally:
        app.state.ready, app.state.pipeline = False, None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="CatVTON Mask-Free API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


async def read_image(upload: UploadFile, name: str) -> Image.Image:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{name} must be an image.")
    content = await upload.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"{name} is too large.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{name} is not a valid image.") from error


def infer(person_image: Image.Image, cloth_image: Image.Image, steps: int, guidance: float, seed: int) -> Image.Image:
    from utils import resize_and_crop, resize_and_padding

    person = resize_and_crop(person_image, (settings.width, settings.height))
    cloth = resize_and_padding(cloth_image, (settings.width, settings.height))
    generator = torch.Generator(device=settings.device).manual_seed(seed)
    return app.state.pipeline(
        image=person,
        condition_image=cloth,
        width=settings.width,
        height=settings.height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )[0]


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not ready.")
    return {"status": "ok"}


@app.post("/v1/try-on", responses={200: {"content": {"image/png": {}}}})
async def try_on(
    request: Request,
    person_image: Annotated[UploadFile, File()],
    cloth_image: Annotated[UploadFile, File()],
    steps: Annotated[int, Form(ge=1, le=100)] = 50,
    guidance_scale: Annotated[float, Form(ge=0, le=10)] = 2.5,
    seed: Annotated[int, Form(ge=0, le=2_147_483_647)] = 42,
) -> Response:
    person, cloth = await read_image(person_image, "person_image"), await read_image(cloth_image, "cloth_image")
    async with request.app.state.lock:
        try:
            result = await run_in_threadpool(infer, person, cloth, steps, guidance_scale, seed)
        except Exception as error:
            logger.exception("CatVTON inference failed")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Image generation failed.") from error
    result_bytes = io.BytesIO()
    result.save(result_bytes, format="PNG")
    return Response(result_bytes.getvalue(), media_type="image/png", headers={"X-Inference-Seed": str(seed)})
