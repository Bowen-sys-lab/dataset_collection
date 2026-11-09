param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Replay,
    [string]$Out = '',
    [string]$Ffmpeg = ''
)

# Runs the existing-image video generator only. No SimulationApp is started.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$replayPath = (Resolve-Path -LiteralPath $Replay).Path
$metadataPath = Join-Path $replayPath 'replay_metadata.json'
$metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
$factoryReplay = $metadata.pipeline -eq 'factory_inspection_offline_v1'
if (-not $Out) {
    # Canonical placement: a new video-<timestamp> next to the replay folder.
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $Out = Join-Path (Split-Path -Parent $replayPath) "video-$stamp"
}
if (Test-Path -LiteralPath $Out) { throw "Output already exists: $Out. Choose a new directory." }
$videoArgs = @('--replay', $replayPath, '--out', $Out)
if ($Ffmpeg) { $videoArgs += @('--ffmpeg', $Ffmpeg) }
$previousPreference = $ErrorActionPreference
$generationLog = [IO.Path]::GetTempFileName()
try {
    $ErrorActionPreference = 'Continue'
    & 'D:\isaacsim\python.bat' -B (Join-Path $PSScriptRoot 'generate_replay_video.py') @videoArgs *> $generationLog
    $videoExitCode = $LASTEXITCODE
    $generationOutput = Get-Content -LiteralPath $generationLog -Raw
    if ($generationOutput) { Write-Host $generationOutput }
    if (Test-Path -LiteralPath $Out -PathType Container) {
        Copy-Item -LiteralPath $generationLog -Destination (Join-Path $Out 'video_generation.log')
    }
} finally {
    $ErrorActionPreference = $previousPreference
    Remove-Item -LiteralPath $generationLog -ErrorAction SilentlyContinue
}
if (-not $factoryReplay) { exit $videoExitCode }
$videoFile = Join-Path $Out 'paired_replay.mp4'
if (-not (Test-Path -LiteralPath $videoFile -PathType Leaf)) {
    throw 'Video generation failed without a video file.'
}
if ($videoExitCode -ne 0 -and $generationOutput -notmatch 'RuntimeError: encoded timestamps do not match replay-relative simulation times') {
    throw 'Video generation failed; only the known factory timestamp-rounding error permits bounded verification.'
}
# Preserve the factory's existing bounded timestamp verification, including
# its legacy fallback when the shared writer produced a video but rejected rounding.
$verificationArgs = @('--replay', $replayPath, '--video', $videoFile,
    '--out', (Join-Path $Out 'video_validation.json'))
if ($Ffmpeg) { $verificationArgs += @('--ffmpeg', $Ffmpeg) }
try {
    $ErrorActionPreference = 'Continue'
    & 'D:\isaacsim\python.bat' -B (Join-Path $PSScriptRoot 'verify_factory_replay_video.py') @verificationArgs
    $verificationExitCode = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference }
exit $verificationExitCode
