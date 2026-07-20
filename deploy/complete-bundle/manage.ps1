$RawArguments = @($args)
$Action = if ($RawArguments.Count -gt 0) { [string]$RawArguments[0] } else { "help" }
$TrainingArgs = if ($RawArguments.Count -gt 1) { @($RawArguments[1..($RawArguments.Count - 1)]) } else { @() }

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$BundleDir = $PSScriptRoot
$DataRoot = Join-Path $BundleDir "data"
$TrainingConfig = Join-Path $BundleDir "training.yaml"
$ImageName = "maiagent-muq-audio:torch2.11.0-cu128"
$ImageArchive = Join-Path $BundleDir "image\maiagent-muq-audio-torch2.11.0-cu128-linux-amd64.tar"
$ImageChecksum = "$ImageArchive.sha256"
$GpuDevices = if ($env:GPU_DEVICES) { $env:GPU_DEVICES } else { "all" }
$ShmSize = if ($env:SHM_SIZE) { $env:SHM_SIZE } else { "16g" }
$TensorBoardPort = if ($env:TENSORBOARD_PORT) { $env:TENSORBOARD_PORT } else { 6006 }

function Stop-WithError {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

function Assert-LastExitCode {
    param([string]$Description)
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "$Description failed with exit code $LASTEXITCODE"
    }
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Stop-WithError "Required command was not found: $Name"
    }
}

function Require-Docker {
    Require-Command "docker"
    & docker info *> $null
    Assert-LastExitCode "Docker daemon check"
}

function Require-LinuxContainers {
    $OsType = (& docker info --format "{{.OSType}}" | Out-String).Trim()
    Assert-LastExitCode "Docker container mode check"
    if ($OsType -ne "linux") {
        Stop-WithError "Docker Desktop must use Linux containers; current OSType is '$OsType'"
    }
}

function Require-Image {
    & docker image inspect $ImageName *> $null
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "Image $ImageName is not loaded. Run .\manage.cmd load"
    }
}

function Test-DataLayout {
    $RequiredPaths = @(
        "datasets\audio_embedding_charts_1000_300_300_train.csv",
        "datasets\audio_embedding_charts_1000_300_300_validation.csv",
        "datasets\audio_embedding_charts_1000_300_300_test.csv",
        "outputs\lancedb\simai_pattern_chunks",
        "outputs\audio_chunks\simai_audio_chunks"
    )
    if (-not (Test-Path -LiteralPath $TrainingConfig -PathType Leaf)) {
        Stop-WithError "Training configuration is missing: $TrainingConfig"
    }
    foreach ($RelativePath in $RequiredPaths) {
        $FullPath = Join-Path $DataRoot $RelativePath
        if (-not (Test-Path -LiteralPath $FullPath)) {
            Stop-WithError "Data path is missing: $FullPath"
        }
    }
}

function Test-ImageArchive {
    if (-not (Test-Path -LiteralPath $ImageArchive -PathType Leaf)) {
        Stop-WithError "Image archive is missing: $ImageArchive"
    }
    if (-not (Test-Path -LiteralPath $ImageChecksum -PathType Leaf)) {
        Stop-WithError "Image checksum is missing: $ImageChecksum"
    }
    $Expected = ((Get-Content -LiteralPath $ImageChecksum -Raw).Trim() -split "\s+")[0].ToLowerInvariant()
    Write-Host "Calculating Docker image archive SHA-256..."
    $Actual = (Get-FileHash -LiteralPath $ImageArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) {
        Stop-WithError "Image archive checksum mismatch. Expected $Expected, got $Actual"
    }
    Write-Host "image_archive_checksum=OK"
}

function Get-CommonDockerArguments {
    return @(
        "run", "--rm", "--init",
        "--shm-size", $ShmSize,
        "--mount", "type=bind,source=$DataRoot,target=/workspace",
        "--mount", "type=bind,source=$TrainingConfig,target=/deploy/training.yaml,readonly",
        "--workdir", "/workspace",
        "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "PYTHONUNBUFFERED=1"
    )
}

function Invoke-TrainingContainer {
    param(
        [bool]$UseGpu,
        [string[]]$ExtraArguments
    )
    Test-DataLayout
    Require-Image
    $DockerArguments = @(Get-CommonDockerArguments)
    if ($UseGpu) {
        $DockerArguments += @("--gpus", $GpuDevices)
    }
    $DockerArguments += @(
        $ImageName,
        "--config", "/deploy/training.yaml",
        "--data-root", "/workspace"
    )
    $DockerArguments += $ExtraArguments
    & docker @DockerArguments
    Assert-LastExitCode "Training container"
}

function Test-GpuRuntime {
    Require-Image
    & docker run --rm --gpus $GpuDevices --entrypoint python $ImageName /opt/maiagent/scripts/check_rtx5090_runtime.py
    Assert-LastExitCode "RTX 5090 CUDA and MuQ runtime check"
}

function Test-Wsl2 {
    Require-Command "wsl.exe"
    & wsl.exe --status
    Assert-LastExitCode "WSL2 status check"
}

function Show-Help {
    @"
Usage: .\manage.cmd ACTION [training CLI overrides]

Actions:
  verify       Verify the bundled image and required data layout.
  load         Verify and load the bundled Docker image.
  doctor       Check WSL2, CUDA, DB joins, and every audio path.
  dry-run      Validate all metadata and files without loading MuQ or CUDA.
  smoke        Run a small one-epoch CUDA training test.
  train        Start full training from training.yaml.
  resume PATH  Resume a checkpoint path relative to the bundled data root.
  tensorboard  Serve run logs on TENSORBOARD_PORT (default 6006).
  config       Print resolved paths and training configuration.
  shell        Open a CUDA shell with bundled data mounted at /workspace.
"@
}

switch ($Action.ToLowerInvariant()) {
    "verify" {
        Test-ImageArchive
        Test-DataLayout
        Write-Host "bundle_verify_complete"
    }
    "load" {
        Require-Docker
        Require-LinuxContainers
        Test-ImageArchive
        & docker image load --input $ImageArchive
        Assert-LastExitCode "Docker image load"
    }
    "doctor" {
        Test-Wsl2
        Require-Docker
        Require-LinuxContainers
        Test-DataLayout
        Test-GpuRuntime
        Invoke-TrainingContainer $false @("--dry-run")
        Write-Host "doctor_complete"
    }
    "dry-run" {
        Require-Docker
        Require-LinuxContainers
        Invoke-TrainingContainer $false (@("--dry-run") + $TrainingArgs)
    }
    "smoke" {
        Require-Docker
        Require-LinuxContainers
        Test-GpuRuntime
        Invoke-TrainingContainer $true (@("--smoke-test") + $TrainingArgs)
    }
    "train" {
        Require-Docker
        Require-LinuxContainers
        Invoke-TrainingContainer $true $TrainingArgs
    }
    "resume" {
        if ($TrainingArgs.Count -eq 0) {
            Stop-WithError "resume requires a checkpoint path relative to the bundled data root"
        }
        $Checkpoint = ([string]$TrainingArgs[0]).Replace("\", "/")
        if ([IO.Path]::IsPathRooted($Checkpoint)) {
            Stop-WithError "Use a checkpoint path relative to the bundled data root"
        }
        $LastSlash = $Checkpoint.LastIndexOf("/")
        if ($LastSlash -lt 1) {
            Stop-WithError "Checkpoint must include its output directory"
        }
        $OutputDirectory = $Checkpoint.Substring(0, $LastSlash)
        $Overrides = @("--resume", $Checkpoint, "--output-dir", $OutputDirectory)
        if ($TrainingArgs.Count -gt 1) {
            $Overrides += $TrainingArgs[1..($TrainingArgs.Count - 1)]
        }
        Require-Docker
        Require-LinuxContainers
        Invoke-TrainingContainer $true $Overrides
    }
    "tensorboard" {
        Require-Docker
        Require-LinuxContainers
        Test-DataLayout
        Require-Image
        $DockerArguments = @(Get-CommonDockerArguments)
        $DockerArguments += @(
            "--publish", "${TensorBoardPort}:6006",
            "--entrypoint", "tensorboard",
            $ImageName,
            "--logdir", "/workspace/outputs/audio_embedding_runs",
            "--host", "0.0.0.0",
            "--port", "6006"
        )
        & docker @DockerArguments
        Assert-LastExitCode "TensorBoard container"
    }
    "config" {
        Write-Host "BUNDLE_DIR=$BundleDir"
        Write-Host "DATA_ROOT=$DataRoot"
        Write-Host "IMAGE_NAME=$ImageName"
        Write-Host "IMAGE_ARCHIVE=$ImageArchive"
        Write-Host "GPU_DEVICES=$GpuDevices"
        Write-Host "SHM_SIZE=$ShmSize"
        Write-Host "TENSORBOARD_PORT=$TensorBoardPort"
        Write-Host ""
        Get-Content -LiteralPath $TrainingConfig
    }
    "shell" {
        Require-Docker
        Require-LinuxContainers
        Test-DataLayout
        Require-Image
        $DockerArguments = @(Get-CommonDockerArguments)
        $DockerArguments += @("-it", "--gpus", $GpuDevices, "--entrypoint", "bash", $ImageName)
        & docker @DockerArguments
        Assert-LastExitCode "Interactive shell"
    }
    { $_ -in @("help", "-h", "--help") } {
        Show-Help
    }
    default {
        Show-Help
        Stop-WithError "Unknown action: $Action"
    }
}
