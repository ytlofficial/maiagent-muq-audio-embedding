# Linux deployment

## Host requirements

- Linux x86-64
- NVIDIA driver compatible with CUDA 12.8
- Docker Engine
- NVIDIA Container Toolkit
- Enough local storage for the Docker image, private data, and checkpoints

The image provides Python, PyTorch 2.11.0, CUDA 12.8 runtime, torchaudio, MuQ,
FFmpeg, LanceDB, librosa, and TensorBoard.

## Configure and run

```bash
cp deploy/linux/deploy.env.example deploy/linux/deploy.env
chmod +x deploy/linux/manage.sh scripts/*.sh
```

Set `DATA_ROOT` in `deploy.env`, then run:

```bash
deploy/linux/manage.sh build
deploy/linux/manage.sh doctor
deploy/linux/manage.sh smoke
deploy/linux/manage.sh train
```

Available actions:

| Action | Purpose |
| --- | --- |
| `build` | Build the pinned CUDA image from this repository |
| `doctor` | Check Docker, CUDA, data layout, DB joins, and audio paths |
| `dry-run` | Validate metadata without loading MuQ or CUDA |
| `smoke` | Run a deterministic small CUDA training pass |
| `train` | Run the configured experiment |
| `resume PATH` | Continue a checkpoint path relative to `DATA_ROOT` |
| `tensorboard` | Serve run logs on the configured port |
| `shell` | Open a CUDA shell with the data root mounted |

Training writes only below `DATA_ROOT/outputs/audio_embedding_runs/`.
