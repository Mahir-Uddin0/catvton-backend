from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import onnx
import torch


def _get_cache_dir() -> str:
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


def _resolve_input_dims(onnx_path: Union[str, Path]) -> Dict[str, List[int | None]]:
    model = onnx.load(str(onnx_path))
    input_dims: Dict[str, List[int | None]] = {}
    for value_info in model.graph.input:
        dims: List[int | None] = []
        tensor_shape = value_info.type.tensor_type.shape
        for dim in tensor_shape.dim:
            if dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        input_dims[value_info.name] = dims
    return input_dims


def _default_shapes(
    onnx_path: Union[str, Path],
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    height: int,
    width: int,
    encoder_sequence: int,
    encoder_hidden_size: int,
) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    input_dims = _resolve_input_dims(onnx_path)
    shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}

    for input_name, dims in input_dims.items():
        if input_name == "sample" and len(dims) == 4:
            channels = dims[1] if dims[1] is not None else 8
            shapes[input_name] = (
                (min_batch, channels, height, width),
                (opt_batch, channels, height, width),
                (max_batch, channels, height, width),
            )
        elif input_name == "timestep" and len(dims) == 1:
            shapes[input_name] = ((1,), (1,), (1,))
        elif input_name == "encoder_hidden_states" and len(dims) == 3:
            shapes[input_name] = (
                (min_batch, encoder_sequence, encoder_hidden_size),
                (opt_batch, encoder_sequence, encoder_hidden_size),
                (max_batch, encoder_sequence, encoder_hidden_size),
            )
        else:
            static_shape = tuple(dim if dim is not None else 1 for dim in dims)
            shapes[input_name] = (static_shape, static_shape, static_shape)

    return shapes


def build_tensorrt_engine_from_onnx(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    fp16: bool = True,
    workspace_size_gb: int = 8,
    min_batch: int = 1,
    opt_batch: int = 2,
    max_batch: int = 4,
    height: int = 128,
    width: int = 96,
    encoder_sequence: int = 1,
    encoder_hidden_size: int = 1,
):
    try:
        import tensorrt as trt
    except ImportError as exc:
        return Path(onnx_path)

    if not torch.cuda.is_available():
        return Path(onnx_path)

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temp_engine_path = engine_path.with_suffix(engine_path.suffix + ".tmp")

    logger = trt.Logger(trt.Logger.INFO)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model at {onnx_path}")

    with trt.Builder(logger) as builder:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with builder.create_network(network_flags) as network:
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, "rb") as onnx_file:
                if not parser.parse(onnx_file.read()):
                    error_messages = [parser.get_error(i) for i in range(parser.num_errors)]
                    raise RuntimeError("TensorRT failed to parse ONNX model:\n" + "\n".join(map(str, error_messages)))

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * 1024**3)
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            shapes = _default_shapes(
                onnx_path=onnx_path,
                min_batch=min_batch,
                opt_batch=opt_batch,
                max_batch=max_batch,
                height=height,
                width=width,
                encoder_sequence=encoder_sequence,
                encoder_hidden_size=encoder_hidden_size,
            )

            for input_index in range(network.num_inputs):
                input_tensor = network.get_input(input_index)
                if input_tensor.name not in shapes:
                    continue
                min_shape, opt_shape, max_shape = shapes[input_tensor.name]
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

            config.add_optimization_profile(profile)

            serialized_engine = None
            if hasattr(builder, "build_serialized_network"):
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    raise RuntimeError("TensorRT returned no serialized engine")
                engine_bytes = bytes(serialized_engine)
            else:
                engine = builder.build_engine(network, config)
                if engine is None:
                    raise RuntimeError("TensorRT failed to build the engine")
                engine_bytes = engine.serialize()

    with open(temp_engine_path, "wb") as engine_file:
        engine_file.write(engine_bytes)
    os.replace(temp_engine_path, engine_path)

    gc.collect()
    return engine_path


def main():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from the cached CatVTON UNet ONNX model and store it in HF_HOME."
    )
    parser.add_argument("--onnx-path", default=None, help="Path to the cached ONNX file. Defaults to HF_HOME/quantization/unet.onnx.")
    parser.add_argument("--engine-path", default=None, help="Path to the engine file. Defaults to HF_HOME/quantization/unet_fp16.engine.")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Enable FP16 optimization when supported by TensorRT.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 optimization.")
    parser.set_defaults(fp16=True)
    parser.add_argument("--workspace-size-gb", type=int, default=8, help="TensorRT workspace size in GB.")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--height", type=int, default=128, help="Latent height used for the TensorRT optimization profile.")
    parser.add_argument("--width", type=int, default=96, help="Latent width used for the TensorRT optimization profile.")
    parser.add_argument("--encoder-sequence", type=int, default=1)
    parser.add_argument("--encoder-hidden-size", type=int, default=1)
    args = parser.parse_args()

    cache_dir = Path(_get_cache_dir())
    onnx_path = Path(args.onnx_path) if args.onnx_path else cache_dir / "quantization" / "unet.onnx"
    engine_path = Path(args.engine_path) if args.engine_path else cache_dir / "quantization" / "unet_fp16.engine"

    built_engine_path = build_tensorrt_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16=args.fp16,
        workspace_size_gb=args.workspace_size_gb,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        height=args.height,
        width=args.width,
        encoder_sequence=args.encoder_sequence,
        encoder_hidden_size=args.encoder_hidden_size,
    )
    print(f"Built TensorRT engine at: {built_engine_path}")


if __name__ == "__main__":
    main()
