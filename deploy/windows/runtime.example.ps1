# Absolute path to the separately transferred private data package.
$DataRoot = "D:\path\to\audio-embedding-data"

$ImageName = "maiagent-muq-audio:torch2.11.0-cu128"
$GpuDevices = "all"
$ShmSize = "20g"
$TensorBoardPort = 6006
$TrainingConfig = "configs\training.example.yaml"

# Docker build options. Weights are downloaded during build, never stored in Git.
$IncludeMuQWeights = "1"
$MuQModelId = "OpenMuQ/MuQ-large-msd-iter"
$PypiIndexUrl = "https://pypi.org/simple"
$PytorchIndexUrl = "https://download.pytorch.org/whl/cu128"
$HfHubOffline = "1"
$TransformersOffline = "1"
