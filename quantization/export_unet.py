from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


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
        str(onnx_path),
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
