param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Run,
    [string]$Out = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath $Run) {
    $runPath = (Resolve-Path -LiteralPath $Run).Path
} else {
    $runPath = Join-Path $projectRoot "validation\trajectory_runs\$Run"
}
$config = Join-Path $projectRoot 'configs\model02_weaklight_scene.yaml'
& (Join-Path $PSScriptRoot 'process_factory_inspection.ps1') -Run $runPath -Config $config -Out $Out
exit $LASTEXITCODE
