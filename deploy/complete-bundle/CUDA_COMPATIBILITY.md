# CUDA 环境验收说明

## 固定环境

- Linux amd64 Docker image
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0 + torchaudio 2.11.0
- cuDNN 9.19 + NCCL 2.28.9
- MuQ 0.1.0 与 `OpenMuQ/MuQ-large-msd-iter` 离线权重

## Windows 主机要求

- Windows 10/11 x86-64
- 支持 CUDA 12.8 的 NVIDIA GPU 与 Windows 驱动
- WSL 2.1.5 或更新
- 最新 Docker Desktop，启用 WSL2 engine 和 Linux containers
- 建议至少 70 GB 可用磁盘；训练输出另计

不要在 WSL 内安装第二套 NVIDIA Linux 显卡驱动。GPU 驱动只安装在 Windows 主机，
CUDA 用户态运行库已经包含在 Docker 镜像中。

## 接收端验收

```powershell
.\manage.cmd verify
.\manage.cmd load
.\manage.cmd doctor
.\manage.cmd smoke
```

`doctor` 必须输出 `"status": "cuda_runtime_ok"`。它会实际执行 CUDA 设备、
PyTorch/torchaudio 版本、FP16（硬件支持时也测 BF16）矩阵计算、AdamW 反向传播、
`8 x 10 秒` MuQ CUDA 前向、三份数据划分、LanceDB 联结和所有音频路径检查。

`smoke` 会再执行一轮小规模端到端训练。两步都通过后运行：

```powershell
.\manage.cmd train
```

如果 `doctor` 报 CUDA 不可用，依次检查 `nvidia-smi`、`wsl --update`、Docker
Desktop 的 WSL2 engine 与 Linux containers，以及 Docker Desktop 是否已重启。
