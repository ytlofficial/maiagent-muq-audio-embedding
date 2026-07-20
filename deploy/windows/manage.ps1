$RawArguments = @($args)
$Action = if ($RawArguments.Count -gt 0) { [string]$RawArguments[0] } else { "help" }
$TrainingArgs = if ($RawArguments.Count -gt 1) { @($RawArguments[1..($RawArguments.Count - 1)]) } else { @() }

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$DeployDir = $PSScriptRoot
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $DeployDir "..\.."))
$RuntimeConfigPath = if ($env:MAIAGENT_DEPLOY_CONFIG) {
    [IO.Path]::GetFullPath($env:MAIAGENT_DEPLOY_CONFIG)
} else {
    Join-Path $DeployDir "runtime.ps1"
}

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

if (-not (Test-Path -LiteralPath $RuntimeConfigPath -PathType Leaf)) {
    Stop-WithError "Create deploy\windows\runtime.ps1 from runtime.example.ps1 and set `$DataRoot"
}
. $RuntimeConfigPath

$RequiredVariables = @(
    "DataRoot", "ImageName", "GpuDevices", "ShmSize", "TensorBoardPort",
    "TrainingConfig", "IncludeMuQWeights", "MuQModelId", "PypiIndexUrl",
    "PytorchIndexUrl", "HfHubOffline", "TransformersOffline"
)
foreach ($VariableName in $RequiredVariables) {
    if (-not (Get-Variable -Name $VariableName -ErrorAction SilentlyContinue)) {
        Stop-WithError "Missing runtime setting: `$${VariableName}"
    }
}

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([IO.Path]::IsPathRooted($PathValue)) {
        return [IO.Path]::GetFullPath($PathValue)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

$TrainingConfigPath = Resolve-RepoPath $TrainingConfig
$script:ResolvedDataRoot = $null

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
        Stop-WithError "Image $ImageName is unavailable. Run .\deploy\windows\manage.cmd build"
    }
}

function Resolve-DataRoot {
    if ([string]::IsNullOrWhiteSpace($DataRoot)) {
        Stop-WithError "Set `$DataRoot in runtime.ps1"
    }
    if (-not [IO.Path]::IsPathRooted($DataRoot)) {
        Stop-WithError "DataRoot must be an absolute Windows path"
    }
    if ($DataRoot.Contains(",")) {
        Stop-WithError "DataRoot cannot contain a comma because Docker --mount uses comma separators"
    }
    if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
        Stop-WithError "DataRoot is not a directory: $DataRoot"
    }
    $script:ResolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
}

function Test-RuntimePaths {
    Resolve-DataRoot
    if (-not (Test-Path -LiteralPath $TrainingConfigPath -PathType Leaf)) {
        Stop-WithError "Training configuration was not found: $TrainingConfigPath"
    }
}

function Build-Image {
    Require-Docker
    Require-LinuxContainers
    & docker build `
        --platform "linux/amd64" `
        --file (Join-Path $RepoRoot "docker\Dockerfile") `
        --tag $ImageName `
        --build-arg "INCLUDE_MUQ_WEIGHTS=$IncludeMuQWeights" `
        --build-arg "MUQ_MODEL_ID=$MuQModelId" `
        --build-arg "PYPI_INDEX_URL=$PypiIndexUrl" `
        --build-arg "PYTORCH_INDEX_URL=$PytorchIndexUrl" `
        $RepoRoot
    Assert-LastExitCode "Docker image build"
}

function Get-CommonDockerArguments {
    return @(
        "run", "--rm", "--init",
        "--shm-size", $ShmSize,
        "--mount", "type=bind,source=$script:ResolvedDataRoot,target=/workspace",
        "--mount", "type=bind,source=$TrainingConfigPath,target=/deploy/training.yaml,readonly",
        "--workdir", "/workspace",
        "--env", "HF_HUB_OFFLINE=$HfHubOffline",
        "--env", "TRANSFORMERS_OFFLINE=$TransformersOffline",
        "--env", "PYTHONUNBUFFERED=1"
    )
}

function Invoke-TrainingContainer {
    param(
        [bool]$UseGpu,
        [string[]]$ExtraArguments
    )
    Test-RuntimePaths
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
    & docker run --rm --gpus $GpuDevices --entrypoint python $ImageName /opt/maiagent/scripts/check_cuda_runtime.py
    Assert-LastExitCode "CUDA and MuQ runtime check"
}

function Test-Wsl2 {
    Require-Command "wsl.exe"
    & wsl.exe --status
    Assert-LastExitCode "WSL2 status check"
}

function Show-Help {
    @"
Usage: .\deploy\windows\manage.cmd ACTION [training CLI overrides]

Actions:
  build        Build the CUDA image from this repository.
  doctor       Check WSL2, Linux containers, CUDA, DB joins, and audio paths.
  dry-run      Validate metadata without loading MuQ or CUDA.
  smoke        Run a small one-epoch CUDA training test.
  train        Start full training from the YAML configuration.
  resume PATH  Resume a checkpoint path relative to DataRoot.
  tensorboard  Serve run logs on TensorBoardPort.
  config       Print resolved runtime and training configuration.
  shell        Open a CUDA shell with DataRoot mounted at /workspace.
  help         Show this message.
"@
}

switch ($Action.ToLowerInvariant()) {
    "build" {
        Build-Image
    }
    "doctor" {
        Test-Wsl2
        Require-Docker
        Require-LinuxContainers
        Test-RuntimePaths
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
            Stop-WithError "resume requires a checkpoint path relative to DataRoot"
        }
        $Checkpoint = ([string]$TrainingArgs[0]).Replace("\", "/")
        if ([IO.Path]::IsPathRooted($Checkpoint)) {
            Stop-WithError "Use a checkpoint path relative to DataRoot"
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
        Test-RuntimePaths
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
        Write-Host "RUNTIME_CONFIG=$RuntimeConfigPath"
        Write-Host "DATA_ROOT=$DataRoot"
        Write-Host "IMAGE_NAME=$ImageName"
        Write-Host "TRAINING_CONFIG=$TrainingConfigPath"
        Write-Host "GPU_DEVICES=$GpuDevices"
        Write-Host "SHM_SIZE=$ShmSize"
        Write-Host "TENSORBOARD_PORT=$TensorBoardPort"
        Write-Host "INCLUDE_MUQ_WEIGHTS=$IncludeMuQWeights"
        Write-Host ""
        Get-Content -LiteralPath $TrainingConfigPath
    }
    "shell" {
        Require-Docker
        Require-LinuxContainers
        Test-RuntimePaths
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
