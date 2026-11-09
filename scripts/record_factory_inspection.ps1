param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$RunId,
    [double]$HorizontalSpeed = 1.5
)

# Factory-only recording entry point. The shared recorder is invoked with the
# factory scene and speed; it is not edited or reconfigured globally.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$config = Join-Path $projectRoot 'configs\factory_inspection_scene.yaml'
& (Join-Path $PSScriptRoot 'record.ps1') -RunId $RunId -Config $config -HorizontalSpeed $HorizontalSpeed
exit $LASTEXITCODE
