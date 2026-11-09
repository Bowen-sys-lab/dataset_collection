param(
    [Parameter(Mandatory = $true, Position = 0)][string]$RunId,
    [ValidateSet('tank','pipe')][string]$Task = 'tank'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$config = Join-Path $projectRoot "configs\model02_v3_${Task}_scene.yaml"
& (Join-Path $PSScriptRoot 'record.ps1') -RunId $RunId -Config $config -HorizontalSpeed 1.5
exit $LASTEXITCODE
