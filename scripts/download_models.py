from pathlib import Path
from huggingface_hub import snapshot_download

CACHE_DIR = str(Path.home() / ".cache" / "huggingface")

repos = [
    "timbrooks/instruct-pix2pix",
    "zhengchong/CatVTON-MaskFree",
    "stabilityai/sd-vae-ft-mse",
]

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

for repo in repos:
    print(f"Downloading {repo}...")
    snapshot_download(
        repo_id=repo,
        cache_dir=CACHE_DIR,
        local_dir_use_symlinks=False,
    )