from __future__ import annotations

from typing import Any

import numpy as np
import torch


_TRITON_TO_NUMPY = {
    "BOOL": np.bool_,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either HTTP JSON metadata or gRPC protobuf metadata."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata_items(metadata: Any, name: str):
    items = _field(metadata, name, [])
    return items or []


def _normalise_url(url: str) -> str:
    return url.removeprefix("http://").removeprefix("https://").removeprefix("grpc://")


class TritonUNet2DConditionModel:
    """Diffusers-compatible U-Net facade backed by a Triton model server."""

    def __init__(
        self,
        client: Any,
        client_module: Any,
        model_name: str,
        model_version: str,
        metadata: Any,
        device: str = "cpu",
    ):
        self.client = client
        self._client_module = client_module
        self.model_name = model_name
        self.model_version = model_version
        self.device = device
        self.metadata = metadata
        self.input_metadata = {
            _field(item, "name"): item
            for item in _metadata_items(metadata, "inputs")
        }
        self.output_metadata = {
            _field(item, "name"): item
            for item in _metadata_items(metadata, "outputs")
        }
        if "sample" not in self.input_metadata:
            raise RuntimeError(
                f"Triton model '{model_name}' has no 'sample' input. "
                f"Available inputs: {sorted(self.input_metadata)}"
            )
        if not self.output_metadata:
            raise RuntimeError(f"Triton model '{model_name}' does not expose an output")
        self.output_name = "noise_pred" if "noise_pred" in self.output_metadata else next(iter(self.output_metadata))

    @classmethod
    def from_server(
        cls,
        url: str = "localhost:8000",
        model_name: str = "unet",
        model_version: str = "",
        protocol: str = "http",
        device: str = "cpu",
        connection_timeout: float = 5.0,
        network_timeout: float = 120.0,
    ) -> "TritonUNet2DConditionModel":
        protocol = protocol.lower()
        if protocol == "http":
            try:
                import tritonclient.http as client_module
            except ImportError as exc:
                raise ImportError(
                    "Triton HTTP support requires 'tritonclient[http]'."
                ) from exc
            client = client_module.InferenceServerClient(
                url=_normalise_url(url),
                verbose=False,
                connection_timeout=connection_timeout,
                network_timeout=network_timeout,
            )
        elif protocol == "grpc":
            try:
                import tritonclient.grpc as client_module
            except ImportError as exc:
                raise ImportError(
                    "Triton gRPC support requires 'tritonclient[grpc]'."
                ) from exc
            client = client_module.InferenceServerClient(
                url=_normalise_url(url),
                verbose=False,
            )
        else:
            raise ValueError("Triton protocol must be 'http' or 'grpc'.")

        if not client.is_server_ready():
            raise RuntimeError(f"Triton server is not ready at {url}")
        if not client.is_model_ready(model_name, model_version=model_version):
            version = model_version or "latest"
            raise RuntimeError(f"Triton model '{model_name}' ({version}) is not ready")

        metadata = client.get_model_metadata(
            model_name,
            model_version=model_version,
        )
        return cls(
            client=client,
            client_module=client_module,
            model_name=model_name,
            model_version=model_version,
            metadata=metadata,
            device=device,
        )

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def close(self):
        close = getattr(self.client, "close", None)
        if close is not None:
            close()

    @staticmethod
    def _tensor_to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.numpy()
        return np.asarray(value)

    def _input_array(self, name: str, sample_array: np.ndarray, timestep: Any, encoder_hidden_states: Any) -> np.ndarray:
        if name == "sample":
            value = sample_array
        elif name == "timestep":
            value = self._tensor_to_numpy(timestep).reshape(-1)
            if value.size == 1 and sample_array.shape[0] > 1:
                value = np.repeat(value, sample_array.shape[0])
        elif name == "encoder_hidden_states":
            if encoder_hidden_states is None:
                value = np.zeros((sample_array.shape[0], 1, 1), dtype=np.float32)
            else:
                value = self._tensor_to_numpy(encoder_hidden_states)
                if value.shape[0] == 1 and sample_array.shape[0] > 1:
                    value = np.repeat(value, sample_array.shape[0], axis=0)
        else:
            raise RuntimeError(f"Unsupported Triton U-Net input '{name}'")

        datatype = _field(self.input_metadata[name], "datatype")
        if datatype == "BF16":
            raise TypeError("BF16 Triton inputs are not supported by NumPy conversion")
        try:
            target_dtype = _TRITON_TO_NUMPY[datatype]
        except KeyError as exc:
            raise TypeError(f"Unsupported Triton datatype '{datatype}' for input '{name}'") from exc
        return np.asarray(value, dtype=target_dtype)

    def __call__(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        sample_tensor = sample if isinstance(sample, torch.Tensor) else torch.as_tensor(sample)
        sample_array = self._tensor_to_numpy(sample_tensor)
        inputs = []
        for name, metadata in self.input_metadata.items():
            array = self._input_array(name, sample_array, timestep, encoder_hidden_states)
            datatype = _field(metadata, "datatype")
            infer_input = self._client_module.InferInput(name, list(array.shape), datatype)
            infer_input.set_data_from_numpy(array)
            inputs.append(infer_input)

        requested_output = self._client_module.InferRequestedOutput(self.output_name)
        result = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=[requested_output],
        )
        output = result.as_numpy(self.output_name)
        if output is None:
            raise RuntimeError(f"Triton response did not contain '{self.output_name}'")
        noise_pred = torch.from_numpy(output).to(
            device=sample_tensor.device,
            dtype=sample_tensor.dtype,
        )
        return (noise_pred,) if not return_dict else {"sample": noise_pred}
