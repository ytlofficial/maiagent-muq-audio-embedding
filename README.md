# MaiAgent MuQ Audio Embedding

This repository trains an audio tower in an existing 512-dimensional chart
embedding space. MuQ remains frozen; only a metadata-conditioned projection
head and the contrastive temperature are optimized.

The repository contains code, tests, example configuration, and deployment
entrypoints only. It intentionally excludes chart files, source audio, audio
chunks, LanceDB contents, split rows, model weights, checkpoints, run logs, and
machine-specific configuration.

## Training flow

```text
split CSV + LanceDB chart/audio/segment tables
  -> strict key joins and song-disjoint split checks
  -> 24 kHz mono waveform
  -> frozen MuQ FP32 frame features
  -> masked mean/max/std pooling
  + difficulty, level, segment id, and six segment scores
  -> normalized 512-D audio embedding
  -> symmetric InfoNCE + cosine distillation
  -> validation retrieval selects best.pt
  -> best.pt evaluates the untouched test split
```

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for the required, data-free schema.

## Repository layout

```text
configs/             editable training configuration
deploy/linux/        Linux Docker management entrypoint
deploy/windows/      Windows + Docker Desktop/WSL2 entrypoint
docker/              CUDA image definition and pinned dependencies
docs/                deployment notes
scripts/             dataset loader, model, trainer, and model cache utility
tests/               metadata, objective, sampler, and training tests
```

## Build the CUDA image

The default build downloads `OpenMuQ/MuQ-large-msd-iter` into the image so it
can train offline. No weight file is stored in Git.

```bash
docker build \
  --file docker/Dockerfile \
  --tag maiagent-muq-audio:torch2.11.0-cu128 \
  --build-arg INCLUDE_MUQ_WEIGHTS=1 \
  .
```

Use `INCLUDE_MUQ_WEIGHTS=0` for a smaller image that downloads the model at
runtime. In that mode, set `model_path: null` and allow Hugging Face access.

## Linux quick start

```bash
cp deploy/linux/deploy.env.example deploy/linux/deploy.env
# Set DATA_ROOT in deploy/linux/deploy.env.

deploy/linux/manage.sh build
deploy/linux/manage.sh doctor
deploy/linux/manage.sh smoke
deploy/linux/manage.sh train
```

Temporary CLI flags override `configs/training.example.yaml`:

```bash
deploy/linux/manage.sh train \
  --output-dir outputs/audio_embedding_runs/experiment-01 \
  --batch-size 8 \
  --learning-rate 5e-4 \
  --epochs 30
```

## Windows quick start

Windows training uses Docker Desktop with the WSL2 Linux-container backend.

```powershell
Copy-Item deploy\windows\runtime.example.ps1 deploy\windows\runtime.ps1
# Set $DataRoot in runtime.ps1.

.\deploy\windows\manage.cmd build
.\deploy\windows\manage.cmd doctor
.\deploy\windows\manage.cmd smoke
.\deploy\windows\manage.cmd train
```

See [docs/linux-deployment.md](docs/linux-deployment.md) and
[docs/windows-wsl2.md](docs/windows-wsl2.md) for host requirements and runtime
commands.

## Build a complete transferable experiment bundle

After building and validating the image, package the image, MuQ weights, split
CSVs, LanceDB, and audio chunks together:

```bash
DATA_SOURCE_ROOT=/absolute/path/to/private-data \
OUTPUT_ROOT=/absolute/path/to/packages \
scripts/package_complete_bundle.sh
```

The command creates an extracted complete bundle and a sibling `-transfer`
directory containing checksum-protected `tar.gz` parts plus Windows/Linux
restore scripts. Private data and generated packages remain ignored by Git.

For one standalone balanced archive, set
`TRANSFER_MODE=single` and a distinct `BUNDLE_NAME`:

```bash
TRANSFER_MODE=single \
BUNDLE_NAME=maiagent-muq-audio-balanced-torch2.11.0-cu128 \
DATA_SOURCE_ROOT=/absolute/path/to/private-data \
OUTPUT_ROOT=/absolute/path/to/packages \
scripts/package_complete_bundle.sh
```

## Local checks

The lightweight tests do not require MuQ or CUDA. Torch-specific tests run when
PyTorch is available.

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

## Model license

MuQ source code and model weights are governed by their upstream licenses. The
`OpenMuQ/MuQ-large-msd-iter` weights are CC BY-NC 4.0; review that license
before building or distributing a weight-bearing image.
