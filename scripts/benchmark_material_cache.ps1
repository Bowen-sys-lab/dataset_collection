param(
    [Parameter(Mandatory = $true)]
    [string]$Out,
    [switch]$ProfilesOnly
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot 'run_isaac_with_clean_env.ps1'
$Probe = Join-Path $ProjectRoot 'tests\isaac_thermal_baseline.py'
$Config = Join-Path $ProjectRoot 'configs\school_vln_scene.yaml'

function Assert-NoIsaacProcess {
    $IsaacProcesses = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(kit|isaac-sim)\.exe$' -or
        ($_.Name -match '^pythonw?\.exe$' -and $_.ExecutablePath -like 'D:\isaacsim\*')
    })
    if ($IsaacProcesses.Count -gt 0) {
        throw ('Isaac is already running; save and close it first. Process IDs: ' +
               (($IsaacProcesses | ForEach-Object ProcessId) -join ', '))
    }
}

Assert-NoIsaacProcess
$OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Output directory already exists; use a new path: $OutputRoot"
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

$Cases = if ($ProfilesOnly) {
    @('off', 'on')
} else {
    @('off', 'on', 'on', 'off', 'off', 'on')
}
for ($CaseIndex = 0; $CaseIndex -lt $Cases.Count; $CaseIndex++) {
    Assert-NoIsaacProcess
    $Mode = $Cases[$CaseIndex]
    $Name = if ($ProfilesOnly) { "profile-$Mode" } else {
        ('perf-{0:D2}-{1}' -f ($CaseIndex + 1), $Mode)
    }
    $Run = Join-Path $OutputRoot $Name
    $RunLog = Join-Path $OutputRoot "$Name.log"
    [string[]]$ProfileArgs = @()
    if ($ProfilesOnly) { $ProfileArgs = @('--profile-only', '--profile-frames', '60') }
    Write-Host "Starting $Name (new process; do not open another Isaac window)"
    & $Launcher $Probe --config $Config --phase baseline --condition "material_cache_$Mode" `
        --rounds 1 --warmup 30 --frames 600 --out $Run @ProfileArgs *> $RunLog
    $ProbeExit = $LASTEXITCODE
    if ($ProbeExit -ne 0) { throw "$Name failed (exit $ProbeExit); see $RunLog" }
    $SummaryPath = Join-Path $Run 'summary.json'
    if (-not (Test-Path -LiteralPath $SummaryPath)) { throw "$Name missing summary.json" }
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    if ($Summary.result -ne 'PASS') { throw "$Name result=$($Summary.result); see $SummaryPath" }
    Write-Host "$Name PASS processing_fps=$($Summary.baseline.rounds[0].processing_fps)"
}
Write-Host "Completed isolated runs: $OutputRoot"
