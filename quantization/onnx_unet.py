from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class OnnxUNet2DConditionModel:
    def __init__(self, session, device="cpu"):
        self.session = session
        self.device = device
        self.input_names = {item.name for item in session.get_inputs()}

    @classmethod
    def from_onnx(cls, onnx_path, device="cpu"):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("OnnxUNet2DConditionModel requires the 'onnxruntime' package") from exc

        session_options = ort.SessionOptions()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if str(device).startswith("cuda") else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(Path(onnx_path)), sess_options=session_options, providers=providers)
        return cls(session=session, device=device)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def __call__(self, sample, timestep, encoder_hidden_states=None, return_dict=False, **kwargs):
        if isinstance(sample, torch.Tensor):
            sample_np = sample.detach().cpu().numpy().astype(np.float32)
            sample_device = sample.device
            sample_dtype = sample.dtype
        else:
            sample_np = np.asarray(sample, dtype=np.float32)
            sample_device = torch.device(self.device)
            sample_dtype = torch.float32

        if isinstance(timestep, torch.Tensor):
            timestep_np = timestep.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            timestep_np = np.asarray(timestep, dtype=np.float32).reshape(-1)

        if encoder_hidden_states is None:
            encoder_hidden_states_np = np.zeros((sample_np.shape[0], 1, 1), dtype=np.float32)
        elif isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states_np = encoder_hidden_states.detach().cpu().numpy().astype(np.float32)
        else:
            encoder_hidden_states_np = np.asarray(encoder_hidden_states, dtype=np.float32)

        outputs = self.session.run(
            None,
            {
                name: value
                for name, value in {
                    "sample": sample_np,
                    "timestep": timestep_np,
                    "encoder_hidden_states": encoder_hidden_states_np,
                }.items()
                if name in self.input_names
            },
        )
        noise_pred = torch.from_numpy(outputs[0]).to(device=sample_device, dtype=sample_dtype)
        return (noise_pred,) if not return_dict else {"sample": noise_pred}
