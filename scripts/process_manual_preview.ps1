param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Run,
    [string]$Out = '',
    [double]$RouteInterval = 1.0
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runPath = if (Test-Path -LiteralPath $Run) { (Resolve-Path -LiteralPath $Run).Path } else { Join-Path $projectRoot "validation\trajectory_runs\$Run" }
# Use the recording's own v3/v4 configuration; never pair an old flight with a new scene.
& (Join-Path $PSScriptRoot 'process_factory_inspection.ps1') -Run $runPath -Out $Out -RouteInterval $RouteInterval -ManualOnly
exit $LASTEXITCODE
