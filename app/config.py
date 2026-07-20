import os
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    catvton_root: Path
    hf_home: Path
    origins: list[str]
    device: str
    dtype: torch.dtype
    width: int
    height: int
    max_upload_bytes: int


def load_settings() -> Settings:
    requested = os.getenv("DEVICE", "auto").lower()

    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cpu":
        device = "cpu"
    else:
        raise RuntimeError("DEVICE must be auto, cpu, or cuda.")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    precision = os.getenv("MIXED_PRECISION", "bf16").lower()

    if device == "cpu" or precision == "no":
        dtype = torch.float32
    elif precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif precision in {"bf16", "fp16"}:
        dtype = torch.float16
    else:
        raise RuntimeError("MIXED_PRECISION must be no, fp16, or bf16.")

    width = int(os.getenv("OUTPUT_WIDTH", "768"))
    height = int(os.getenv("OUTPUT_HEIGHT", "1024"))

    if min(width, height) <= 0 or width % 8 or height % 8:
        raise RuntimeError(
            "OUTPUT_WIDTH and OUTPUT_HEIGHT must be positive multiples of 8."
        )

    origins = [
        item.strip()
        for item in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000"
        ).split(",")
        if item.strip()
    ]

    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser().resolve()

    return Settings(
        catvton_root=Path(
            os.getenv("CATVTON_ROOT", ROOT / "CatVTON")
        ).expanduser().resolve(),
        hf_home=hf_home,
        origins=origins,
        device=device,
        dtype=dtype,
        width=width,
        height=height,
        max_upload_bytes=int(
            os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
        ),
    )