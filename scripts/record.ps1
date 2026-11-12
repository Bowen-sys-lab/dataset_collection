param(
    [Parameter(Position = 0)]
    [string]$RunId = '',
    [string]$Config = '',
    [double]$HorizontalSpeed = 5,
    [ValidateSet('city')]
    [string]$Scene = '',
    [string]$Target = '',
    [ValidateRange(1,3)]
    [int]$InstructionVariant = 1,
    [string]$StartGroup = '',
    [string]$Episode = '',
    [switch]$CheckOnly,
    [switch]$CollisionUndo,
    [ValidateRange(0,120)]
    [int]$SmokeFrames = 0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

if ($Episode) {
    if (-not $Scene) { throw 'Episode requires -Scene.' }
    if ($Target -or $StartGroup -or $Config -or $PSBoundParameters.ContainsKey('InstructionVariant') -or
        $PSBoundParameters.ContainsKey('HorizontalSpeed') -or $RunId) {
        throw 'Episode specifies its own task/start/control contract. Do not combine it with overrides.'
    }
    if ($Scene -eq 'city') {
        # City v2 episodes C01..C20 are task-table rows, not the pilot30 factory/forest manifest.
        if ($Episode -notmatch '^C[0-9]{2}$') {
            throw 'City episodes use C01..C20. See configs/collection_episodes/city_v2.json'
        }
        if ($SmokeFrames -gt 0) {
            throw 'City v2 episodes do not support -SmokeFrames yet.'
        }
        $manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'configs\collection_episodes\city_v2.json') `
            -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = $manifest.episodes | Where-Object { $_.episode -eq $Episode } | Select-Object -First 1
        if (-not $row) { throw "Unknown city Episode '$Episode'." }
        $backend = @{ Task = $row.task_id }
        if ($CheckOnly) { $backend.CheckOnly = $true }
        if ($PSBoundParameters.ContainsKey('CollisionUndo')) { $backend.CollisionUndo = [bool]$CollisionUndo }
        & (Join-Path $PSScriptRoot 'record_city_v2_task.ps1') @backend
        if (-not $?) { exit 1 }
        exit 0
    }
    throw 'Only City episodes are included in this handoff. Use -Scene city -Episode C01..C20.'
}

# Scene-aware dispatch: scene data comes from configs/collection_profiles/<scene>.yaml
# and the scene plan it points to; this entry only validates and forwards.
if ($Scene) {
    $resolver = Join-Path $PSScriptRoot 'collection_profile.py'
    $resolved = $null
    $resolverArgs = @('-X','utf8',$resolver,'--scene',$Scene,'--variant',[string]$InstructionVariant)
    if ($Target) { $resolverArgs += @('--target',$Target) }
    if ($StartGroup) { $resolverArgs += @('--start-group',$StartGroup) }
    foreach ($python in @('python','D:\isaacsim\python.bat')) {
        try {
            $raw = & $python @resolverArgs 2>$null
            $last = ($raw | Select-Object -Last 1)
            if ($last) { $resolved = $last | ConvertFrom-Json }
            if ($resolved) { break }
        } catch {
            $resolved = $null
        }
    }
    if (-not $resolved) { throw "Cannot resolve scene '$Scene': python with PyYAML is required (configs/collection_profiles/$Scene.yaml)" }
    if (-not $resolved.ok) { throw $resolved.error }
    $Target = $resolved.target
    $StartGroup = $resolved.start_group
    $backendArgs = @{
        Target = $Target
        InstructionVariant = $InstructionVariant
        StartGroup = $StartGroup
    }
    if ($RunId) { $backendArgs.RunId = $RunId }
    if ($CheckOnly) { $backendArgs.CheckOnly = $true }
    if ($PSBoundParameters.ContainsKey('CollisionUndo')) { $backendArgs.CollisionUndo = [bool]$CollisionUndo }
    if ($SmokeFrames -gt 0) { $backendArgs.SmokeFrames = $SmokeFrames }
    Write-Host "[record] scene=$($resolved.scene) target=$Target variant=$InstructionVariant start=$StartGroup plan=$($resolved.plan)"
    & $resolved.entry @backendArgs
    exit $LASTEXITCODE
}

# Legacy generic path: config-driven recording, unchanged.
if (-not $Config) { $Config = Join-Path $projectRoot 'configs\school_vln_scene.yaml' }
if (-not $RunId) { $RunId = Get-Date -Format 'yyyyMMdd-HHmmss-fff' }
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') {
    throw 'RunId must contain only letters, digits, underscores or hyphens.'
}
$runPath = Join-Path $projectRoot "validation\trajectory_runs\$RunId"
if (Test-Path -LiteralPath $runPath) { throw "Run already exists: $runPath. Choose a new RunId." }
$shellPath = (Get-Process -Id $PID).Path
Write-Host "Recording: $RunId. Click Drone Control; T marks success. Close the preview when finished."
$scriptArgs = @('--config', $Config, '--preview-mode', 'live', '--record-trajectory',
                '--run-id', $RunId, '--horizontal-speed',
                $HorizontalSpeed.ToString([Globalization.CultureInfo]::InvariantCulture), '--no-save')
if ($PSBoundParameters.ContainsKey('CollisionUndo')) {
    $scriptArgs += $(if ($CollisionUndo) { '--collision-undo' } else { '--no-collision-undo' })
}
# Pass an explicit array within a child session so Python's --options cannot
# bind to PowerShell common parameters. Quote every path/value as a literal.
$clean = (Join-Path $PSScriptRoot 'run_isaac_with_clean_env.ps1').Replace("'", "''")
$script = (Join-Path $PSScriptRoot 'start_crazyflie_dual_camera_view.py').Replace("'", "''")
$quotedArgs = ($scriptArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ','
$childCommand = "& '$clean' -ScriptPath '$script' -ScriptArgs @($quotedArgs)"
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCommand))
& $shellPath -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedCommand
exit $LASTEXITCODE
