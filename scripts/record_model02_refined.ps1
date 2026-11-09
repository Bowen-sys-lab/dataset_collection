param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$RunId
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$config = Join-Path $projectRoot 'configs\model02_refined_scene.yaml'
& (Join-Path $PSScriptRoot 'record.ps1') -RunId $RunId -Config $config -HorizontalSpeed 1.5
exit $LASTEXITCODE

