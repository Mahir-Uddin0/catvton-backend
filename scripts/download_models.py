from huggingface_hub import snapshot_download

cache_dir = "/root/.cache/huggingface"

repos = [
    "timbrooks/instruct-pix2pix",
    "zhengchong/CatVTON-MaskFree",
    "stabilityai/sd-vae-ft-mse",
]

for repo in repos:
    print(f"Downloading {repo}...")
    snapshot_download(
        repo_id=repo,
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,
    )