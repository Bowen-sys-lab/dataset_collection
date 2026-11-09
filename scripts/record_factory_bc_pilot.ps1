param(
    [Parameter(Position = 0)][string]$RunId = '',
    [switch]$CheckOnly,
    [switch]$CollisionUndo,
    [ValidateRange(0, 120)][int]$SmokeFrames = 0,
    [ValidateSet('tank','pipe')][string]$Target = 'tank',
    [ValidateRange(1,3)][int]$InstructionVariant = 1,
    [ValidateSet('A','B','D','E','F')][string]$StartGroup = 'A'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunId) {
    if ($SmokeFrames -gt 0) {
        $RunId = 'factory-bc-probe-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    } else {
        for ($number = 1; $number -le 999; $number++) {
            $candidate = 'factory-bc-pilot-{0:D3}' -f $number
            $used = (Test-Path -LiteralPath (Join-Path $projectRoot "validation\trajectory_runs\$candidate")) -or
                    (Test-Path -LiteralPath (Join-Path $projectRoot "validation\factory-bc-pilot\previews\$candidate")) -or
                    (Test-Path -LiteralPath (Join-Path $projectRoot "validation\factory-bc-pilot\launches\$candidate.json"))
            if (-not $used) { $RunId = $candidate; break }
        }
        if (-not $RunId) { throw 'All pilot numbers 001..999 are used. Supply a new RunId.' }
    }
}
$pythonArgs = @('--run-id', $RunId,'--target',$Target,'--instruction-variant',[string]$InstructionVariant,'--start-group',$StartGroup)
if ($PSBoundParameters.ContainsKey('CollisionUndo')) {
    $pythonArgs += $(if ($CollisionUndo) { '--collision-undo' } else { '--no-collision-undo' })
}
if ($CheckOnly) { $pythonArgs += '--check-only' }
if ($SmokeFrames -gt 0) { $pythonArgs += @('--smoke-frames', [string]$SmokeFrames) }
$clean = (Join-Path $PSScriptRoot 'run_isaac_with_clean_env.ps1').Replace("'", "''")
$script = (Join-Path $PSScriptRoot 'factory_bc_pilot.py').Replace("'", "''")
$literalArgs = ($pythonArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ','
$childCommand = "& '$clean' -ScriptPath '$script' -ScriptArgs @($literalArgs)"
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCommand))
$shellPath = (Get-Process -Id $PID).Path
& $shellPath -NoProfile -OutputFormat Text -ExecutionPolicy Bypass -EncodedCommand $encodedCommand
exit $LASTEXITCODE
