import os
from pathlib import Path

import torch
from PIL import Image

from app.config import load_settings
from CatVTON.model.pipeline import CatVTONPix2PixPipeline
from CatVTON.utils import resize_and_crop, resize_and_padding

settings = load_settings()
os.environ.setdefault("HF_HOME", str(settings.hf_home))

# CPU settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
torch.set_num_threads(max(1, os.cpu_count() or 1))

# Small resolution + few steps: validates the whole pipeline, not image quality.
WIDTH = 768
HEIGHT = 1024
STEPS = 1
GUIDANCE_SCALE = 2.5  # Avoids classifier-free-guidance's doubled CPU work.
SEED = 42

PERSON = Path("inputs/person.jpg")
CLOTH = Path("inputs/cloth.jpg")
OUTPUT = Path("outputs/maskfree_cpu_smoke_test.png")

pipeline = CatVTONPix2PixPipeline(
    base_ckpt="timbrooks/instruct-pix2pix",
    attn_ckpt="zhengchong/CatVTON-MaskFree",
    attn_ckpt_version="mix-48k-1024",
    weight_dtype=DTYPE,
    device=DEVICE,
    use_tf32=False,            # CUDA-only optimization
    skip_safety_check=True,    # Avoids loading an unneeded CLIP safety model
)

person = resize_and_crop(Image.open(PERSON).convert("RGB"), (WIDTH, HEIGHT))
cloth = resize_and_padding(Image.open(CLOTH).convert("RGB"), (WIDTH, HEIGHT))

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

result = pipeline(
    image=person,
    condition_image=cloth,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator,
)[0]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT)
print(f"Saved CPU smoke-test result to: {OUTPUT}")