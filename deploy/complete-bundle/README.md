# MaiAgent MuQ 完整训练包

本目录同时包含 CUDA/PyTorch/MuQ 依赖镜像和训练所需数据。接收电脑解压后不需要
另外下载 Python 包、模型权重、谱面向量或音频切片。

## 包内结构

```text
README.md
manifest.json
training.yaml
manage.sh
manage.cmd
manage.ps1
image/
  maiagent-muq-audio-torch2.11.0-cu128-linux-amd64.tar
  maiagent-muq-audio-torch2.11.0-cu128-linux-amd64.tar.sha256
data/
  datasets/
  outputs/lancedb/simai_pattern_chunks/
  outputs/audio_chunks/simai_audio_chunks/
```

镜像固定为 PyTorch 2.11.0、torchvision 0.26.0、torchaudio 2.11.0、CUDA
12.8、cuDNN 9、MuQ 0.1.0，并内置 `OpenMuQ/MuQ-large-msd-iter` 权重。
详细兼容性边界与验收项目见 `CUDA_COMPATIBILITY.md`。

默认采用稳健高吞吐训练档：30 epochs、batch 32、validation batch 64、6 workers、
learning rate 0.001。它不会把 GPU 显存压满；若目标机显存较小，可临时降到
`--batch-size 16 --validation-batch-size 32`。

## Windows 10/11 + NVIDIA GPU

安装支持 CUDA 12.8 的 NVIDIA Windows 驱动、WSL `2.1.5+` 和 Docker Desktop。
启用 WSL2 backend 与 Linux containers，然后运行：

```powershell
.\manage.cmd verify
.\manage.cmd load
.\manage.cmd doctor
.\manage.cmd smoke
.\manage.cmd train
```

## Linux + NVIDIA GPU

安装 Docker Engine 和 NVIDIA Container Toolkit，然后运行：

```bash
chmod +x manage.sh
./manage.sh verify
./manage.sh load
./manage.sh doctor
./manage.sh smoke
./manage.sh train
```

`doctor` 会检查 CUDA GPU、PyTorch/CUDA 版本、FP16/BF16 矩阵计算、反向传播、
MuQ CUDA 前向、三份 split、LanceDB 三表联结以及每个音频路径。`smoke` 会真正跑
一轮小规模 CUDA 训练。只有这两步通过后才建议开始完整训练。

## 参数修改

训练参数统一修改 `training.yaml`。也可以临时覆盖：

```powershell
.\manage.cmd train --batch-size 8 --learning-rate 5e-4 --epochs 30
```

输出保存在 `data/outputs/audio_embedding_runs/`。续训示例：

```powershell
.\manage.cmd resume outputs/audio_embedding_runs/muq_baseline/last.pt --epochs 40
```

若显存不足，先降低 `batch_size` 和 `validation_batch_size`。MuQ 权重采用
CC BY-NC 4.0，带权重镜像应按该许可用于研究和非商业场景。
