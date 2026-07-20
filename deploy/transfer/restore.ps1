param(
    [string]$Destination = ".",
    [switch]$KeepArchive
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$TransferDir = $PSScriptRoot
$ChecksumFile = Join-Path $TransferDir "parts.sha256"
$ArchiveChecksumFile = Join-Path $TransferDir "archive.sha256"

function Stop-WithError {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

$ResolvedDestination = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $ResolvedDestination -Force | Out-Null

if (Test-Path -LiteralPath $ArchiveChecksumFile -PathType Leaf) {
    $Fields = ((Get-Content -LiteralPath $ArchiveChecksumFile -Raw).Trim() -split "\s+", 2)
    if ($Fields.Count -ne 2) {
        Stop-WithError "Invalid archive checksum file: $ArchiveChecksumFile"
    }
    $ArchivePath = Join-Path $TransferDir $Fields[1].Trim()
    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        Stop-WithError "Missing archive: $ArchivePath"
    }
    Write-Host "Calculating standalone archive SHA-256..."
    $Actual = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Fields[0].ToLowerInvariant()) {
        Stop-WithError "Checksum mismatch: $ArchivePath"
    }
    $TemporaryArchive = $false
} else {
    if (-not (Test-Path -LiteralPath $ChecksumFile -PathType Leaf)) {
        Stop-WithError "Missing archive.sha256 or parts.sha256"
    }
    $Lines = @(Get-Content -LiteralPath $ChecksumFile | Where-Object { $_.Trim() })
    if ($Lines.Count -eq 0) {
        Stop-WithError "Checksum file is empty"
    }
    $Parts = @()
    $Index = 0
    foreach ($Line in $Lines) {
        $Fields = $Line.Trim() -split "\s+", 2
        if ($Fields.Count -ne 2) {
            Stop-WithError "Invalid checksum line: $Line"
        }
        $PartPath = Join-Path $TransferDir $Fields[1].Trim()
        if (-not (Test-Path -LiteralPath $PartPath -PathType Leaf)) {
            Stop-WithError "Missing archive part: $PartPath"
        }
        $Index += 1
        Write-Progress -Activity "Verifying archive parts" -Status "$Index / $($Lines.Count)" -PercentComplete (($Index * 100) / $Lines.Count)
        $Actual = (Get-FileHash -LiteralPath $PartPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Fields[0].ToLowerInvariant()) {
            Stop-WithError "Checksum mismatch: $PartPath"
        }
        $Parts += Get-Item -LiteralPath $PartPath
    }
    Write-Progress -Activity "Verifying archive parts" -Completed

    $ArchiveName = ([IO.Path]::GetFileName($Parts[0].Name) -split "\.part-")[0]
    $ArchivePath = Join-Path $ResolvedDestination $ArchiveName
    Write-Host "Joining $($Parts.Count) verified parts into $ArchivePath"
    $Output = [IO.File]::Open($ArchivePath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $Index = 0
        foreach ($Part in ($Parts | Sort-Object Name)) {
            $Index += 1
            Write-Progress -Activity "Joining archive" -Status "$Index / $($Parts.Count)" -PercentComplete (($Index * 100) / $Parts.Count)
            $Input = [IO.File]::OpenRead($Part.FullName)
            try {
                $Input.CopyTo($Output)
            } finally {
                $Input.Dispose()
            }
        }
    } finally {
        $Output.Dispose()
        Write-Progress -Activity "Joining archive" -Completed
    }
    $TemporaryArchive = $true
}

if (-not (Get-Command "tar.exe" -ErrorAction SilentlyContinue)) {
    Stop-WithError "Windows tar.exe is required to extract the complete bundle"
}
& tar.exe -xzf $ArchivePath -C $ResolvedDestination
if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Archive extraction failed with exit code $LASTEXITCODE"
}

if ($TemporaryArchive -and -not $KeepArchive) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
Write-Host "restored_to=$ResolvedDestination"
