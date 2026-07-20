#!/usr/bin/env python3
"""Validate the generic CUDA, PyTorch, and MuQ runtime path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
import torchaudio
import torchvision


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_driver_version() -> str | None:
    try:
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
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None


def static_report() -> dict[str, object]:
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
    require(bool(compiled_arches), "PyTorch reports no compiled CUDA architectures")
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_compiled": torch._C._cuda_getCompiledVersion(),
        "cudnn": torch.backends.cudnn.version(),
        "compiled_arches": compiled_arches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    report = static_report()
    if args.static_only:
        report["status"] = "cuda_static_compatibility_ok"
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    require(torch.cuda.is_available(), "CUDA is unavailable inside the Docker container")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    torch.manual_seed(2026)
    math_results: dict[str, bool] = {}
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        left = torch.randn((1024, 1024), device=device, dtype=dtype)
        right = torch.randn((1024, 1024), device=device, dtype=dtype)
        product = left @ right
        torch.cuda.synchronize()
        name = str(dtype).removeprefix("torch.")
        math_results[name] = bool(torch.isfinite(product).all())
        require(math_results[name], f"non-finite {dtype} matmul")

    layer = torch.nn.Linear(1024, 512, device=device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    inputs = torch.randn((8, 1024), device=device)
    loss = layer(inputs).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    require(bool(torch.isfinite(loss)), "non-finite CUDA backward loss")

    from muq import MuQ

    profile_batch_size = int(os.environ.get("CUDA_PROFILE_BATCH_SIZE", "8"))
    profile_audio_seconds = int(os.environ.get("CUDA_PROFILE_AUDIO_SECONDS", "10"))
    require(profile_batch_size > 0, "CUDA_PROFILE_BATCH_SIZE must be positive")
    require(profile_audio_seconds > 0, "CUDA_PROFILE_AUDIO_SECONDS must be positive")
    model_path = Path(os.environ.get("MUQ_MODEL_PATH", "/opt/models/MuQ-large-msd-iter"))
    require(model_path.is_dir(), f"embedded MuQ model is missing: {model_path}")
    model = MuQ.from_pretrained(str(model_path)).to(device).eval()
    waveform = torch.zeros(
        (profile_batch_size, profile_audio_seconds * 24_000),
        device=device,
    )
    attention_mask = torch.ones_like(waveform, dtype=torch.long)
    with torch.inference_mode():
        output = model(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
    torch.cuda.synchronize()
    require(bool(torch.isfinite(output).all()), "MuQ CUDA output contains non-finite values")

    report.update(
        {
            "status": "cuda_runtime_ok",
            "device": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "vram_bytes": properties.total_memory,
            "driver": read_driver_version(),
            "math": math_results,
            "backward_loss": float(loss.detach().cpu()),
            "muq_output_shape": list(output.shape),
            "muq_output_finite": True,
            "profile_batch_size": profile_batch_size,
            "profile_audio_seconds": profile_audio_seconds,
        }
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
