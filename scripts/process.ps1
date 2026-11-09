param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Run,
    [string]$Config = '',
    [string]$Out = '',
    [double]$SampleInterval = 1.0
)

# Offline workflow only: each Isaac replay runs in a fresh child process;
# quality checking and indexing start after that process exits. Video is manual.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath $Run) {
    $runPath = (Resolve-Path -LiteralPath $Run).Path
} else {
    $runPath = Join-Path $projectRoot "validation\trajectory_runs\$Run"
}
if (Test-Path -LiteralPath $runPath -PathType Container) {
    $trajectoryPath = Join-Path $runPath 'trajectory.jsonl'
} else {
    $trajectoryPath = $runPath
}
$trajectoryPath = (Resolve-Path -LiteralPath $trajectoryPath).Path
$runFolder = Split-Path -Parent $trajectoryPath
$runId = Split-Path -Leaf $runFolder
$episodeMetadataPath = Join-Path $runFolder 'run_metadata.json'
if (Test-Path -LiteralPath $episodeMetadataPath) {
    $episodeRecording = Get-Content -LiteralPath $episodeMetadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $episodeSelection = $episodeRecording.task.environment_evidence.collection_episode
    if ($episodeSelection) {
        if ($Config) { throw 'Episode replay uses its recorded configuration; omit -Config.' }
        $episodeInterval = if ($PSBoundParameters.ContainsKey('SampleInterval')) { $SampleInterval } else { 1.0 }
        $episodeArgs = @('-B','-m','scripts.process_collection_episode','--run',$runFolder,
                         '--interval',$episodeInterval.ToString([Globalization.CultureInfo]::InvariantCulture))
        if ($Out) { $episodeArgs += @('--out',$Out) }
        Push-Location $projectRoot
        $episodePreference = $ErrorActionPreference
        try { $ErrorActionPreference='Continue'; & 'D:\isaacsim\python.bat' @episodeArgs; $episodeExit = $LASTEXITCODE }
        finally { $ErrorActionPreference=$episodePreference; Pop-Location }
        exit $episodeExit
    }
}
if (-not $Config) {
    $metadataPath = Join-Path $runFolder 'run_metadata.json'
    if (-not (Test-Path -LiteralPath $metadataPath)) {
        throw 'No run_metadata.json. Supply -Config explicitly for this legacy trajectory.'
    }
    $recording = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $Config = $recording.scene_config_path
    if (-not $Config) { throw 'No recorded scene configuration. Supply -Config explicitly.' }
}
$Config = (Resolve-Path -LiteralPath $Config).Path
if ($SampleInterval -le 0 -or [double]::IsNaN($SampleInterval) -or [double]::IsInfinity($SampleInterval)) {
    throw 'SampleInterval must be a finite positive number of simulation seconds.'
}
# Canonical contract: validation/episodes/<RunId>/<processing timestamp>/
if (-not $Out) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $Out = Join-Path $projectRoot "validation\episodes\$runId\$stamp"
}
if (Test-Path -LiteralPath $Out) { throw "Output already exists: $Out. Choose a new directory." }
$outputRoot = (New-Item -ItemType Directory -Path $Out).FullName
$replayPath = Join-Path $outputRoot 'replay'
$logPath = Join-Path $outputRoot 'logs'
$datasetPath = Join-Path $outputRoot 'dataset'
# replay/ is created by its Python entrypoint with
# exist_ok=False; only the orchestration directories are created here.
foreach ($directory in @($logPath, $datasetPath)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}
$statusPath = Join-Path $outputRoot 'pipeline_status.json'
$sceneId = 'generic'
if ($Config -match 'city') { $sceneId = 'city' }
elseif ($Config -match 'factory|model02') { $sceneId = 'factory' }
$script:status = [ordered]@{
    schema_version = 1
    run_id = $runId
    scene_id = $sceneId
    backend = 'generic_image_replay'
    config_path = $Config
    out_root = $outputRoot
    started_at = (Get-Date).ToString('o')
    finished_at = $null
    status = 'RUNNING'
    error = $null
    video = $null
    stages = (New-Object System.Collections.ArrayList)
}
$script:statusPath = $statusPath
function Write-PipelineStatus {
    $json = $script:status | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($script:statusPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}
function Add-Stage([string]$name, [string]$state, [string]$log, $artifacts, [string]$reason = '') {
    [void]$script:status.stages.Add([ordered]@{
        name = $name; status = $state; log = $log; artifacts = $artifacts; reason = $reason
    })
}
Add-Stage 'video' 'SKIPPED' $null $null 'manual_only'
Write-PipelineStatus

$shellPath = (Get-Process -Id $PID).Path
$pythonBat = 'D:\isaacsim\python.bat'
$intervalArg = $SampleInterval.ToString([Globalization.CultureInfo]::InvariantCulture)
$clean = (Join-Path $PSScriptRoot 'run_isaac_with_clean_env.ps1').Replace("'", "''")
$script = (Join-Path $PSScriptRoot 'replay_trajectory_highres.py').Replace("'", "''")
$childLauncher = Join-Path $PSScriptRoot 'invoke_logged_powershell.ps1'

try {
    Write-Host '[1/3] Offline RGB 256x256 + Thermal 256x256 replay...'
    $scriptArgs = @('--config', $Config, '--trajectory', $trajectoryPath, '--out', $replayPath,
                    '--sample-interval', $intervalArg)
    $quotedArgs = ($scriptArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ','
    $childCommand = "& '$clean' -ScriptPath '$script' -ScriptArgs @($quotedArgs)"
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCommand))
    $replayLog = Join-Path $logPath 'replay.log'
    $replayErrorLog = Join-Path $logPath 'replay.stderr.log'
    $replayExitCode = & $childLauncher -ShellPath $shellPath -EncodedCommand $encodedCommand `
        -StdoutPath $replayLog -StderrPath $replayErrorLog
    if ($replayExitCode -ne 0) {
        throw "Replay failed (exit $replayExitCode). See $replayLog, $replayErrorLog and replay\failure.txt."
    }
    $failurePath = Join-Path $replayPath 'failure.txt'
    if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
        throw "Replay wrote a failure report. See $replayLog and $failurePath."
    }
    $replayMetadataPath = Join-Path $replayPath 'replay_metadata.json'
    if (-not (Test-Path -LiteralPath $replayMetadataPath -PathType Leaf)) {
        throw "Replay exited without replay_metadata.json. See $replayLog."
    }
    try {
        $replayMetadata = Get-Content -LiteralPath $replayMetadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw "Replay metadata is invalid. See $replayMetadataPath and $replayLog."
    }
    if ($replayMetadata.result -ne 'PASS') {
        throw "Replay did not report PASS (result='$($replayMetadata.result)'). See $replayMetadataPath."
    }
    Add-Stage 'replay' 'PASS' $replayLog $replayPath
    Write-PipelineStatus

    Write-Host '[2/3] Checking files, arrays, timestamps, poses and task provenance...'
    $validationLog = Join-Path $logPath 'validation.log'
    & $pythonBat -B (Join-Path $PSScriptRoot 'validate_replay_outputs.py') --replay $replayPath *> $validationLog
    if ($LASTEXITCODE -ne 0) {
        throw "Validation failed. See $validationLog and replay\validation_summary.json."
    }
    Add-Stage 'validation' 'PASS' $validationLog (Join-Path $replayPath 'validation_summary.json')
    Write-PipelineStatus

    Write-Host '[3/3] Writing the per-processing data index...'
    $indexLog = Join-Path $logPath 'index.log'
    if (Test-Path -LiteralPath (Join-Path $runFolder 'run_metadata.json')) {
        Push-Location $projectRoot
        try {
            & $pythonBat -B -m scripts.episode_dataset --run $runFolder --replay $replayPath --out $datasetPath *> $indexLog
            $indexExitCode = $LASTEXITCODE
        } finally { Pop-Location }
        if ($indexExitCode -ne 0) { throw "Frame index failed. See $indexLog." }
    }
    # Per-run data index (this processing pass only); cross-run publishing is a
    # separate validation/datasets/<DatasetId> step.
    $frameCount = $replayMetadata.frames
    $index = [ordered]@{
        schema_version = 1
        run_id = $runId
        scene_id = $sceneId
        processed_at = (Get-Date).ToString('o')
        source = [ordered]@{
            trajectory = $trajectoryPath
            run_metadata = (Join-Path $runFolder 'run_metadata.json')
            config_snapshot = (Join-Path $runFolder 'config_snapshot')
        }
        artifacts = [ordered]@{
            replay = $replayPath
            video = $null
        }
        frame_count = $frameCount
        sample_interval_s = $SampleInterval
        source_sample_period_s = $replayMetadata.trajectory_provenance.sample_period_s
        capture_contract = $replayMetadata.capture_contract
        replay_capture_contract = $replayMetadata.replay_capture_contract
        rgb_resolution = $replayMetadata.rgb_resolution
        thermal_resolution = $replayMetadata.thermal_resolution
        task_acceptance = 'NOT_EVALUATED'
        note = 'Per-processing index only; not a certified training-ready manifest.'
    }
    $indexJson = $index | ConvertTo-Json -Depth 10
    [IO.File]::WriteAllText((Join-Path $datasetPath 'index.json'), $indexJson,
        (New-Object System.Text.UTF8Encoding($false)))
    Add-Stage 'dataset_index' 'PASS' $null (Join-Path $datasetPath 'index.json')

    $script:status.status = 'COMPLETE'
    $script:status.finished_at = (Get-Date).ToString('o')
    Write-PipelineStatus
    Write-Host "Done (pilot/development data): $outputRoot"
    Write-Host 'Video generation skipped (manual_only). Use video.ps1 with the replay directory when needed.'
} catch {
    $script:status.status = 'FAILED'
    $script:status.finished_at = (Get-Date).ToString('o')
    $script:status.error = $_.Exception.Message
    Write-PipelineStatus
    throw
}
