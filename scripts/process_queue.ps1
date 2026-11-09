param(
    [string[]]$Runs = @(),
    [string]$RunList = '',
    [string]$Queue = '',
    [switch]$Resume,
    [switch]$CheckOnly,
    [double]$SampleInterval = 1.0
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$arguments = @('-B', '-X', 'utf8', '-m', 'scripts.process_queue')
if ($Runs.Count) { $arguments += '--runs'; $arguments += $Runs }
if ($RunList) { $arguments += @('--run-list', $RunList) }
if ($Queue) { $arguments += @('--queue', $Queue) }
if ($Resume) { $arguments += '--resume' }
if ($CheckOnly) { $arguments += '--check-only' }
if ($PSBoundParameters.ContainsKey('SampleInterval')) {
    $arguments += @('--interval', $SampleInterval.ToString([Globalization.CultureInfo]::InvariantCulture))
}
# The queue itself is CPU-only; each existing process.ps1 starts its Isaac child.
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = 'D:\isaacsim\python.bat' }
Push-Location $root
try {
    $ErrorActionPreference = 'Continue'
    & $python @arguments
    $queueExit = $LASTEXITCODE
} finally { Pop-Location }
exit $queueExit
