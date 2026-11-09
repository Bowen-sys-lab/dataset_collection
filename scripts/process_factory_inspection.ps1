param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Run,
    [string]$Config = '',
    [string]$Out = '',
    [double]$RouteInterval = 1.0,
    [switch]$ManualOnly
)

# Factory-only offline workflow. The Isaac child process is isolated from the
# caller; validation and evaluation run after it exits. Video is manual. This script never
# changes the source trajectory, stage, configuration, or shared replay code.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath $Run) {
    $runPath = (Resolve-Path -LiteralPath $Run).Path
} else {
    $runPath = Join-Path $projectRoot "validation\factory-inspection\$Run"
}
if (Test-Path -LiteralPath $runPath -PathType Container) {
    $trajectoryPath = Join-Path $runPath 'trajectory.jsonl'
} else {
    $trajectoryPath = $runPath
}
$trajectoryPath = (Resolve-Path -LiteralPath $trajectoryPath).Path
$runFolder = Split-Path -Parent $trajectoryPath
if (-not $Config) {
    $metadataPath = Join-Path $runFolder 'run_metadata.json'
    if (-not (Test-Path -LiteralPath $metadataPath)) { throw 'No run_metadata.json. Supply -Config explicitly.' }
    $recording = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $Config = $recording.scene_config_path
    if (-not $Config) { throw 'No recorded scene configuration. Supply -Config explicitly.' }
}
$Config = (Resolve-Path -LiteralPath $Config).Path
if ($RouteInterval -le 0 -or [double]::IsNaN($RouteInterval) -or [double]::IsInfinity($RouteInterval)) {
    throw 'RouteInterval must be a finite positive number of simulation seconds.'
}
if (-not $Out) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $Out = Join-Path $projectRoot "validation\factory-inspection\episodes\$((Split-Path -Leaf $runFolder))-$stamp"
}
if (Test-Path -LiteralPath $Out) { throw "Output already exists: $Out" }
$outputRoot = (New-Item -ItemType Directory -Path $Out).FullName
$replayPath = Join-Path $outputRoot 'replay'
$evaluationPath = Join-Path $outputRoot 'evaluation'
$shellPath = (Get-Process -Id $PID).Path
$pythonBat = 'D:\isaacsim\python.bat'
$intervalArg = $RouteInterval.ToString([Globalization.CultureInfo]::InvariantCulture)

function Invoke-FactoryPython {
    param([string[]]$Arguments,[string]$LogPath)
    $previousPreference=$ErrorActionPreference
    try {
        # Windows PowerShell 5 otherwise turns the first stderr line into a
        # terminating NativeCommandError before the traceback reaches the log.
        $ErrorActionPreference='Continue'
        & $pythonBat -B @Arguments *> $LogPath
        $pythonExitCode=$LASTEXITCODE
    } finally { $ErrorActionPreference=$previousPreference }
    return [int]$pythonExitCode
}

Write-Host '[1/3] Factory offline RGB 256x256 + thermal 256x256 replay...'
$scriptArgs = @('--config', $Config, '--trajectory', $trajectoryPath, '--out', $replayPath,
                '--route-interval', $intervalArg)
if ($ManualOnly) { $scriptArgs += '--manual-only' }
$clean = (Join-Path $PSScriptRoot 'run_isaac_with_clean_env.ps1').Replace("'", "''")
$script = (Join-Path $PSScriptRoot 'replay_factory_inspection.py').Replace("'", "''")
$quotedArgs = ($scriptArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ','
$childCommand = "& '$clean' -ScriptPath '$script' -ScriptArgs @($quotedArgs)"
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCommand))
$replayExitCode = & (Join-Path $PSScriptRoot 'invoke_logged_powershell.ps1') -ShellPath $shellPath `
    -EncodedCommand $encodedCommand -StdoutPath (Join-Path $outputRoot 'replay.log') `
    -StderrPath (Join-Path $outputRoot 'replay.stderr.log')
if ($replayExitCode -ne 0) { throw "Factory replay failed (exit $replayExitCode). See $outputRoot\replay.stderr.log and $replayPath\failure.txt." }
$gpuFailure=Select-String -LiteralPath (Join-Path $outputRoot 'replay.log'),(Join-Path $outputRoot 'replay.stderr.log') `
    -SimpleMatch -Pattern 'Out of GPU memory','ERROR_OUT_OF_DEVICE_MEMORY' -Quiet
if($gpuFailure -contains $true){throw "GPU memory exhausted during replay. Partial frames are not accepted. See $outputRoot\replay.stderr.log"}
if(-not(Test-Path -LiteralPath (Join-Path $replayPath 'replay_metadata.json'))){
    throw "Replay ended without its final metadata; output is incomplete. See $outputRoot\replay.log and replay.stderr.log."
}

Write-Host '[2/3] Checking factory replay outputs, evidence, poses and finite arrays...'
$validationExitCode=Invoke-FactoryPython -Arguments @((Join-Path $PSScriptRoot 'replay_factory_inspection.py'),'--validate-only',$replayPath) `
    -LogPath (Join-Path $outputRoot 'validation.log')
if ($validationExitCode -ne 0) { throw "Factory replay validation failed. See $outputRoot\validation.log." }

if ($ManualOnly) {
    Write-Host '[3/3] Copying recorded manual confirmation; no automatic inspection rules...'
    $evaluationExitCode=Invoke-FactoryPython -Arguments @((Join-Path $PSScriptRoot 'replay_factory_inspection.py'),'--trajectory',$trajectoryPath,
        '--manual-summary','--out',$evaluationPath) -LogPath (Join-Path $outputRoot 'evaluation.log')
} else {
    Write-Host '[3/3] Evaluating factory route and visual hold evidence...'
    $evaluationExitCode=Invoke-FactoryPython -Arguments @((Join-Path $PSScriptRoot 'evaluate_factory_inspection.py'),'--run',$runFolder,'--config',$Config,
        '--out',$evaluationPath,'--visual-evidence',(Join-Path $replayPath 'visual_evidence.jsonl')) -LogPath (Join-Path $outputRoot 'evaluation.log')
}
if ($evaluationExitCode -ne 0) { throw "Factory evaluation failed. See evaluation.log." }

Write-Host "Done (pilot/development data): $outputRoot"
Write-Host 'Video generation skipped (manual_only). Use video.ps1 with the replay directory when needed.'
