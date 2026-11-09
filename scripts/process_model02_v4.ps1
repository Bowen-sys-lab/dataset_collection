param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Run,
    [ValidateSet('fault','detail')][string]$View = 'fault',
    [string]$Out = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runPath = if (Test-Path -LiteralPath $Run) { (Resolve-Path -LiteralPath $Run).Path } else { Join-Path $projectRoot "validation\trajectory_runs\$Run" }
$sceneName = if ($View -eq 'detail') { 'model02_v4_detail_scene.yaml' } else { 'model02_v4_tank_scene.yaml' }
& (Join-Path $PSScriptRoot 'process_factory_inspection.ps1') -Run $runPath -Config (Join-Path $projectRoot "configs\$sceneName") -Out $Out
exit $LASTEXITCODE
