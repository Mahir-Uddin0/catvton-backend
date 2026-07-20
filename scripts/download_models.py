from pathlib import Path

from huggingface_hub import snapshot_download

from app.config import load_settings


settings = load_settings()

CACHE_DIR = settings.hf_home

print(f"HF_HOME: {CACHE_DIR}")

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

repos = [
    "timbrooks/instruct-pix2pix",
    "zhengchong/CatVTON-MaskFree",
    "stabilityai/sd-vae-ft-mse",
]

for repo in repos:
    print(f"Downloading {repo}...")

    path = snapshot_download(
        repo_id=repo,
        cache_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded to: {path}")