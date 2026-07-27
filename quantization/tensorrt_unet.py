from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _trt_dtype_to_torch(trt_dtype):
    name = str(trt_dtype).lower()
    if "float16" in name or "fp16" in name:
        return torch.float16
    if "float32" in name or "fp32" in name:
        return torch.float32
    if "int32" in name:
        return torch.int32
    if "int8" in name:
        return torch.int8
    return torch.float32


class TensorRTUNet2DConditionModel:
    def __init__(self, backend, device="cpu"):
        self.backend = backend
        self.device = device
        self.input_names = set(backend.get("input_names", {"sample", "timestep", "encoder_hidden_states"}))
        self._mode = backend.get("mode", "onnx")

    @classmethod
    def from_engine(cls, engine_path, device="cpu", fallback_to_onnx=True):
        engine_path = Path(engine_path)
        if str(device).startswith("cuda"):
            try:
                import tensorrt as trt

                if engine_path.exists():
                    logger = trt.Logger(trt.Logger.INFO)
                    runtime = trt.Runtime(logger)
                    with open(engine_path, "rb") as engine_file:
                        engine_bytes = engine_file.read()
                    engine = runtime.deserialize_cuda_engine(engine_bytes)
                    if engine is None:
                        raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")
                    context = engine.create_execution_context()
                    if context is None:
                        raise RuntimeError("Failed to create TensorRT execution context")
                    input_names = []
                    output_names = []
                    if hasattr(engine, "num_io_tensors"):
                        for index in range(engine.num_io_tensors):
                            tensor_name = engine.get_tensor_name(index)
                            mode = engine.get_tensor_mode(tensor_name)
                            mode_name = getattr(mode, "name", str(mode))
                            if mode_name.endswith("INPUT"):
                                input_names.append(tensor_name)
                            else:
                                output_names.append(tensor_name)
                    else:
                        for index in range(engine.num_bindings):
                            name = engine.get_binding_name(index)
                            if engine.binding_is_input(index):
                                input_names.append(name)
                            else:
                                output_names.append(name)
                    backend = {
                        "mode": "tensorrt",
                        "engine": engine,
                        "context": context,
                        "input_names": input_names,
                        "output_names": output_names,
                    }
                    return cls(backend=backend, device=device)
            except Exception:
                if not fallback_to_onnx:
                    raise

        if fallback_to_onnx:
            candidate_onnx_paths = [
                engine_path.with_name("unet.onnx"),
                engine_path.with_suffix(".onnx"),
                engine_path,
            ]
            onnx_path = next((path for path in candidate_onnx_paths if path.exists()), candidate_onnx_paths[0])
            return cls.from_onnx(onnx_path, device=device)
        raise FileNotFoundError(f"Missing TensorRT engine at {engine_path}")

    @classmethod
    def from_onnx(cls, onnx_path, device="cpu"):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("TensorRTUNet2DConditionModel requires either 'tensorrt' or 'onnxruntime'") from exc

        session_options = ort.SessionOptions()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if str(device).startswith("cuda") else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(Path(onnx_path)), sess_options=session_options, providers=providers)
        backend = {
            "mode": "onnx",
            "session": session,
            "input_names": [item.name for item in session.get_inputs()],
        }
        return cls(backend=backend, device=device)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def _prepare_inputs(self, sample, timestep, encoder_hidden_states):
        if isinstance(sample, torch.Tensor):
            sample_tensor = sample
            sample_device = sample.device
            sample_dtype = sample.dtype
        else:
            sample_tensor = torch.as_tensor(sample)
            sample_device = torch.device(self.device)
            sample_dtype = torch.float32

        timestep_tensor = timestep if isinstance(timestep, torch.Tensor) else torch.as_tensor(timestep)
        timestep_tensor = timestep_tensor.reshape(-1).contiguous()

        if encoder_hidden_states is None:
            encoder_hidden_states_tensor = torch.zeros(sample_tensor.shape[0], 1, 1, device=sample_device, dtype=sample_dtype)
        else:
            encoder_hidden_states_tensor = encoder_hidden_states if isinstance(encoder_hidden_states, torch.Tensor) else torch.as_tensor(encoder_hidden_states)

        return sample_tensor.to(device=sample_device).contiguous(), timestep_tensor.to(device=sample_device).contiguous(), encoder_hidden_states_tensor.to(device=sample_device).contiguous()

    def __call__(self, sample, timestep, encoder_hidden_states=None, return_dict=False, **kwargs):
        sample_tensor, timestep_tensor, encoder_hidden_states_tensor = self._prepare_inputs(sample, timestep, encoder_hidden_states)
        input_dtype = sample_tensor.dtype

        if self._mode == "tensorrt":
            backend = self.backend
            engine = backend["engine"]
            context = backend["context"]
            input_names = backend["input_names"]
            output_names = backend["output_names"]
            input_map = {
                "sample": sample_tensor,
                "timestep": timestep_tensor,
                "encoder_hidden_states": encoder_hidden_states_tensor,
            }
            input_map = {name: tensor for name, tensor in input_map.items() if name in input_names}
            output_tensors = {}
            input_dtypes = {name: _trt_dtype_to_torch(engine.get_tensor_dtype(name)) for name in input_names}
            output_dtypes = {name: _trt_dtype_to_torch(engine.get_tensor_dtype(name)) for name in output_names}
            for name, tensor in input_map.items():
                if tensor.dtype != input_dtypes[name]:
                    tensor = tensor.to(dtype=input_dtypes[name])
                    input_map[name] = tensor
                context.set_input_shape(name, tuple(tensor.shape))
                context.set_tensor_address(name, int(tensor.data_ptr()))
            for output_name in output_names:
                output_shape = tuple(context.get_tensor_shape(output_name))
                output_tensors[output_name] = torch.empty(output_shape, device=sample_tensor.device, dtype=output_dtypes[output_name])
                context.set_tensor_address(output_name, int(output_tensors[output_name].data_ptr()))
            stream = torch.cuda.current_stream(sample_tensor.device)
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            noise_pred = output_tensors[output_names[0]].to(dtype=input_dtype)
        else:
            session = self.backend["session"]
            input_feed = {}
            if "sample" in self.input_names:
                input_feed["sample"] = sample_tensor.detach().cpu().numpy().astype(np.float32)
            if "timestep" in self.input_names:
                input_feed["timestep"] = timestep_tensor.detach().cpu().numpy().astype(np.float32)
            if "encoder_hidden_states" in self.input_names:
                input_feed["encoder_hidden_states"] = encoder_hidden_states_tensor.detach().cpu().numpy().astype(np.float32)
            outputs = session.run(None, input_feed)
            noise_pred = torch.from_numpy(outputs[0]).to(device=sample_tensor.device, dtype=input_dtype)

        return (noise_pred,) if not return_dict else {"sample": noise_pred}
