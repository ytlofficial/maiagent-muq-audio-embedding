# RTX 5090 / Windows 验收说明

## 固定环境

- Linux amd64 Docker image
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0 + torchaudio 2.11.0
- cuDNN 9.19 + NCCL 2.28.9
- MuQ 0.1.0 与 `OpenMuQ/MuQ-large-msd-iter` 离线权重
- PyTorch 原生架构：`sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`

RTX 5090 的计算能力是 12.0，对应 `sm_120`。本镜像包含原生 `sm_120` 内核。

## Windows 主机要求

- Windows 10/11 x86-64；优先使用最新版 Windows 11
- NVIDIA Windows driver 最低 570.65，建议 580.88 或更新
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

`doctor` 必须输出 `"status": "rtx5090_runtime_ok"`。它会实际执行：

1. RTX 5090 型号、32 位驱动版本与计算能力 12.0 检查。
2. PyTorch `sm_120` 原生架构检查。
3. FP16 与 BF16 CUDA 矩阵乘法及有限值检查。
4. CUDA AdamW 反向传播与参数更新。
5. 内置 MuQ 权重的一秒音频 CUDA 前向及有限值检查。
6. 三份数据划分、LanceDB 联结和所有音频路径检查。

`smoke` 会再执行一轮小规模端到端训练。两步都通过后运行：

```powershell
.\manage.cmd train
```

如果 `doctor` 报 CUDA 不可用，依次检查 `nvidia-smi`、`wsl --update`、Docker
Desktop 的 WSL2 engine 与 Linux containers，以及 Docker Desktop 是否已重启。
