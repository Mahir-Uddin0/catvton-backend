from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Union

import torch

from CatVTON.model.attn_processor import SkipAttnProcessor
from CatVTON.model.utils import get_trainable_module, init_adapter
from diffusers import UNet2DConditionModel

# import get_cache_dir
def _get_cache_dir():
    try:
        from app.config import load_settings

        settings = load_settings()
        cache_dir = str(settings.hf_home)
    except Exception:
        cache_dir = os.environ.get("HF_HOME")

    if not cache_dir:
        cache_dir = os.path.join(str(Path.home()), ".cache", "huggingface", "hub")

    os.environ["HF_HOME"] = cache_dir
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    return cache_dir


def export_unet_to_onnx(
    unet: torch.nn.Module,
    onnx_path: Union[str, Path],
    sample_input_shape=None,
    encoder_hidden_states_shape=None,
    opset_version: int = 17,
):
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("export_unet_to_onnx requires the 'onnx' package") from exc

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx_path = onnx_path.with_suffix(onnx_path.suffix + ".tmp")

    if sample_input_shape is None:
        in_channels = getattr(getattr(unet, "config", None), "in_channels", None)
        if in_channels is None:
            raise ValueError("Unable to infer UNet input channels for ONNX export")
        sample_input_shape = (2, int(in_channels), 128, 96)

    class _ExportWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module):
            super().__init__()
            self.module = module

        def forward(self, sample, timestep, encoder_hidden_states=None):
            return self.module(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

    wrapper = _ExportWrapper(unet).eval()
    sample = torch.randn(*sample_input_shape, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.float32)
    if encoder_hidden_states_shape is None:
        encoder_hidden_states = torch.zeros(sample_input_shape[0], 1, 1, dtype=torch.float32)
    else:
        encoder_hidden_states = torch.randn(*encoder_hidden_states_shape, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (sample, timestep, encoder_hidden_states),
        str(temp_onnx_path),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        dynamic_axes={
            "sample": {0: "batch", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "noise_pred": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset_version,
    )
    os.replace(temp_onnx_path, onnx_path)


def export_pretrained_unet_to_onnx(
    base_ckpt: str,
    attn_ckpt: str,
    attn_ckpt_version: str,
    onnx_path: Union[str, Path],
    device: str = "cuda",
    weight_dtype: torch.dtype = torch.float32,
    cache_dir: Union[str, Path, None] = None,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if cache_dir is None:
        cache_dir = _get_cache_dir()
    cache_dir = Path(cache_dir)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    pytorch_unet = UNet2DConditionModel.from_pretrained(
        base_ckpt,
        subfolder="unet",
        cache_dir=str(cache_dir),
    ).to(device, dtype=weight_dtype)
    init_adapter(pytorch_unet, cross_attn_cls=SkipAttnProcessor)
    attn_modules = get_trainable_module(pytorch_unet, "attention")

    from accelerate import load_checkpoint_in_model

    if os.path.exists(attn_ckpt):
        load_checkpoint_in_model(attn_modules, os.path.join(attn_ckpt, attn_ckpt_version, "attention"))
    else:
        from huggingface_hub import snapshot_download

        repo_path = snapshot_download(repo_id=attn_ckpt, cache_dir=str(cache_dir))
        load_checkpoint_in_model(attn_modules, os.path.join(repo_path, attn_ckpt_version, "attention"))

    export_unet_to_onnx(pytorch_unet, onnx_path)

    del attn_modules
    del pytorch_unet
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Export the CatVTON UNet to ONNX and store it in the cache directory.")
    parser.add_argument("--base-ckpt", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--attn-ckpt", default="zhengchong/CatVTON-MaskFree")
    parser.add_argument("--attn-ckpt-version", default="mix-48k-1024")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device for loading/exporting the PyTorch UNet.")
    parser.add_argument("--onnx-path", default=None)
    args = parser.parse_args()

    cache_dir = Path(_get_cache_dir())
    onnx_path = Path(args.onnx_path) if args.onnx_path else cache_dir / "quantization" / "unet.onnx"
    export_pretrained_unet_to_onnx(
        base_ckpt=args.base_ckpt,
        attn_ckpt=args.attn_ckpt,
        attn_ckpt_version=args.attn_ckpt_version,
        onnx_path=onnx_path,
        device=args.device,
        cache_dir=cache_dir,
    )
    print(f"Exported ONNX UNet to: {onnx_path}")


if __name__ == "__main__":
    main()
