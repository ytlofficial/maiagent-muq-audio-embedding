# MaiAgent MuQ Audio Embedding

This repository builds the Simai chart-side training inputs and trains an audio
tower in the resulting 512-dimensional chart embedding space. MuQ remains
frozen; only a metadata-conditioned projection head and the contrastive
temperature are optimized.

The repository contains code, tests, example configuration, and deployment
entrypoints only. It intentionally excludes chart files, source audio, audio
chunks, LanceDB contents, split rows, model weights, checkpoints, run logs, and
machine-specific configuration.

## Training flow

```text
maidata / exported Simai charts
  -> measure timeline compiler and note counter
  -> density-aware segmentation and six-dimensional segment scores
  -> four-measure structural chart embeddings
  -> optional TruncatedSVD projection to 512-D teacher vectors
  -> LanceDB chart/audio/segment tables
  -> song-disjoint split CSVs
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

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for the required, data-free runtime
schema and [docs/usage.md](docs/usage.md) for the preprocessing scripts.

## Repository layout

```text
configs/             editable training configuration
deploy/linux/        Linux Docker management entrypoint
deploy/windows/      Windows + Docker Desktop/WSL2 entrypoint
docker/              CUDA image definition and pinned dependencies
docs/                deployment notes, scoring formulas, preprocessing usage
scripts/             Simai preprocessing, LanceDB builders, dataset loader, trainer
simai_machine_readable_spec/
                     Simai grammar/spec artifacts used by the parser
tests/               parser, embedding, metadata, objective, sampler, and training tests
```

## Preprocessing overview

The upstream chart algorithm is stored in `scripts/` and is intentionally
data-free:

- `simai_measure_compiler.py` compiles Simai timeline slots into measures and
  time ranges.
- `simai_note_counter.py`, `simai_measure_density.py`, and
  `simai_density_segmenter.py` parse notes, compute density curves, and build
  five density-aware segments.
- `simai_global_six_dimension_table.py` and `simai_segment_scorer.py` compute
  the `note`, `peak`, `charge`, `slide`, `handtrip`, and `tricky` scores.
- `simai_pattern_embedding.py` builds the raw structural four-measure chart
  embeddings.
- `rebuild_full_pattern_embedding_lancedb_svd512.py` projects raw chart
  embeddings into the 512-D teacher space used by audio training.
- `build_segment_chunk_lancedb.py`, `build_segment_chunk_audio_lancedb.py`,
  and `split_audio_embedding_charts.py` assemble the training LanceDB tables
  and song-disjoint CSV splits.

The generated datasets, source audio, LanceDB files, and checkpoints stay out
of Git by design; the algorithms and contracts needed to recreate them are in
this repository.

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

The lightweight parser, scoring, and preprocessing tests do not require MuQ or
CUDA. LanceDB writers are tested through pure row-building helpers; full table
writes need the Docker/runtime dependencies. Torch-specific tests run when
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
