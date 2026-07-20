# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG INCLUDE_MUQ_WEIGHTS=1
ARG MUQ_MODEL_ID=OpenMuQ/MuQ-large-msd-iter
ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    MUQ_MODEL_PATH=/opt/models/MuQ-large-msd-iter \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv --system-site-packages /opt/venv

ENV PATH=/opt/venv/bin:${PATH}

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
        torchvision==0.26.0 torchaudio==2.11.0 \
    && python -m pip install --index-url "${PYPI_INDEX_URL}" -r /tmp/requirements.txt \
    && python -c "import torch, torchvision, torchaudio, muq, lancedb; assert torch.__version__.startswith('2.11.0'); assert torchvision.__version__.startswith('0.26.0'); assert torchaudio.__version__.startswith('2.11.0'); assert torch.version.cuda == '12.8'; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.version.cuda)"

COPY scripts/cache_muq_model.py /opt/maiagent/scripts/cache_muq_model.py
RUN if [ "${INCLUDE_MUQ_WEIGHTS}" = "1" ]; then \
        python /opt/maiagent/scripts/cache_muq_model.py \
            --model-id "${MUQ_MODEL_ID}" \
            --output-dir "${MUQ_MODEL_PATH}"; \
    fi

COPY scripts/audio_embedding_dataset.py /opt/maiagent/scripts/audio_embedding_dataset.py
COPY scripts/check_cuda_runtime.py /opt/maiagent/scripts/check_cuda_runtime.py
COPY scripts/muq_audio_training.py /opt/maiagent/scripts/muq_audio_training.py
COPY scripts/train_muq_audio_embedding.py /opt/maiagent/scripts/train_muq_audio_embedding.py

WORKDIR /workspace
ENTRYPOINT ["python", "/opt/maiagent/scripts/train_muq_audio_embedding.py"]
CMD ["--device", "cuda"]
