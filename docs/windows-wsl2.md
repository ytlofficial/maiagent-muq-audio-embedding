# Windows deployment with WSL2

CUDA training runs in a Linux container through Docker Desktop's WSL2 backend.
Native Windows containers are not supported by this image.

## Host requirements

- Windows 10 or 11 x86-64 with WSL 2.1.5 or newer
- NVIDIA GPU and Windows driver compatible with CUDA 12.8
- Docker Desktop using the WSL2 engine and Linux containers
- A local SSD for the private data package and checkpoints

## Configure and run

```powershell
Copy-Item deploy\windows\runtime.example.ps1 deploy\windows\runtime.ps1
```

Set `$DataRoot` in `runtime.ps1`, then run:

```powershell
.\deploy\windows\manage.cmd build
.\deploy\windows\manage.cmd doctor
.\deploy\windows\manage.cmd smoke
.\deploy\windows\manage.cmd train
```

Use a path relative to `$DataRoot` when resuming:

```powershell
.\deploy\windows\manage.cmd resume outputs/audio_embedding_runs/run-01/last.pt --epochs 40
```

The manager verifies WSL2, Linux-container mode, CUDA availability, FP16/BF16 CUDA
math, backward propagation, a real MuQ CUDA forward pass, required private data
paths, LanceDB joins, and every resolved audio file before full training starts.
