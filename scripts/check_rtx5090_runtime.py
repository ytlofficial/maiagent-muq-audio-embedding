#!/usr/bin/env python3
"""Validate the complete CUDA/MuQ path on a Windows-hosted RTX 5090."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch
import torchaudio
import torchvision


MINIMUM_CUDA_128_WINDOWS_DRIVER = (570, 65)
RECOMMENDED_WINDOWS_DRIVER = (580, 88)


def parse_driver_version(value: str) -> tuple[int, ...]:
    numbers = tuple(int(part) for part in re.findall(r"\d+", value))
    if len(numbers) < 2:
        raise RuntimeError(f"unable to parse NVIDIA driver version: {value!r}")
    return numbers


def read_driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not versions:
        raise RuntimeError("nvidia-smi returned no NVIDIA driver version")
    return versions[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    require(torch.__version__.startswith("2.11.0"), f"unexpected torch: {torch.__version__}")
    require(
        torchvision.__version__.startswith("0.26.0"),
        f"unexpected torchvision: {torchvision.__version__}",
    )
    require(
        torchaudio.__version__.startswith("2.11.0"),
        f"unexpected torchaudio: {torchaudio.__version__}",
    )
    require(torch.version.cuda == "12.8", f"unexpected CUDA runtime: {torch.version.cuda}")

    compiled_arches = str(torch._C._cuda_getArchFlags()).split()
    require("sm_120" in compiled_arches, f"PyTorch has no native sm_120 kernel: {compiled_arches}")
    if args.static_only:
        print(
            json.dumps(
                {
                    "status": "rtx5090_static_compatibility_ok",
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "torchaudio": torchaudio.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cuda_compiled": torch._C._cuda_getCompiledVersion(),
                    "cudnn": torch.backends.cudnn.version(),
                    "compiled_arches": compiled_arches,
                    "required_compute_capability": "12.0",
                    "required_native_arch": "sm_120",
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    require(torch.cuda.is_available(), "CUDA is unavailable inside the Docker container")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    device_name = properties.name
    capability = (properties.major, properties.minor)
    require("RTX 5090" in device_name.upper(), f"expected RTX 5090, found {device_name}")
    require(capability == (12, 0), f"expected compute capability 12.0, found {capability}")

    driver_version = read_driver_version()
    parsed_driver = parse_driver_version(driver_version)
    require(
        parsed_driver >= MINIMUM_CUDA_128_WINDOWS_DRIVER,
        "Windows NVIDIA driver must be at least 570.65 for CUDA 12.8",
    )

    torch.manual_seed(5090)
    math_results: dict[str, bool] = {}
    for dtype in (torch.float16, torch.bfloat16):
        left = torch.randn((1024, 1024), device=device, dtype=dtype)
        right = torch.randn((1024, 1024), device=device, dtype=dtype)
        product = left @ right
        torch.cuda.synchronize()
        math_results[str(dtype).removeprefix("torch.")] = bool(torch.isfinite(product).all())
        require(math_results[str(dtype).removeprefix("torch.")], f"non-finite {dtype} matmul")

    layer = torch.nn.Linear(1024, 512, device=device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    inputs = torch.randn((8, 1024), device=device)
    loss = layer(inputs).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    require(bool(torch.isfinite(loss)), "non-finite CUDA backward loss")

    from muq import MuQ

    model_path = Path(os.environ.get("MUQ_MODEL_PATH", "/opt/models/MuQ-large-msd-iter"))
    require(model_path.is_dir(), f"embedded MuQ model is missing: {model_path}")
    model = MuQ.from_pretrained(str(model_path)).to(device).eval()
    waveform = torch.zeros((1, 24_000), device=device)
    attention_mask = torch.ones_like(waveform, dtype=torch.long)
    with torch.inference_mode():
        output = model(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
    torch.cuda.synchronize()
    require(bool(torch.isfinite(output).all()), "MuQ CUDA output contains non-finite values")

    report = {
        "status": "rtx5090_runtime_ok",
        "device": device_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "vram_bytes": properties.total_memory,
        "driver": driver_version,
        "driver_recommended": parsed_driver >= RECOMMENDED_WINDOWS_DRIVER,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "compiled_arches": compiled_arches,
        "math": math_results,
        "backward_loss": float(loss.detach().cpu()),
        "muq_output_shape": list(output.shape),
        "muq_output_finite": True,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
