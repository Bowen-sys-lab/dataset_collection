param([string]$Config = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonBat = 'D:\isaacsim\python.bat'
if (-not (Test-Path -LiteralPath $pythonBat)) { throw "Install the specified Isaac Sim at D:\isaacsim first." }
$reportPath = Join-Path $projectRoot ('validation\handoff\environment-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.json')
$checkArgs = @('-B','-m','scripts.dataset_handoff','check','--out',$reportPath)
if ($Config) { $checkArgs += @('--config',$Config) }
Push-Location $projectRoot
try { & $pythonBat @checkArgs; $code=$LASTEXITCODE }
finally { Pop-Location }
Write-Host "Software check report: $reportPath"
Write-Host 'This does not certify GPU rendering, scene assets, human flight or task success.'
exit $code
