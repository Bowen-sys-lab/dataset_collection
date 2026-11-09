param(
    [Parameter(Position = 0)][string]$RunId = '',
    [ValidateSet('fault','detail')][string]$View = 'fault',
    [ValidateRange(0.1,20.0)][double]$HorizontalSpeed = 10.0
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sceneName = if ($View -eq 'detail') { 'model02_v5_detail_scene.yaml' } else { 'model02_v5_tank_scene.yaml' }
$config = Join-Path $projectRoot "configs\$sceneName"
& (Join-Path $PSScriptRoot 'record.ps1') -RunId $RunId -Config $config -HorizontalSpeed $HorizontalSpeed
exit $LASTEXITCODE
