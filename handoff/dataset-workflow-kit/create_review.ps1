param(
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string]$Batch
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') { throw 'Invalid RunId.' }
$runPath=Join-Path $projectRoot "validation\trajectory_runs\$RunId"
$outputPath=Join-Path $projectRoot ("validation\annotations\$RunId\" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
Push-Location $projectRoot
try {
    & 'D:\isaacsim\python.bat' -B -m scripts.dataset_handoff init-review --run $runPath --batch $Batch --out $outputPath
    $code=$LASTEXITCODE
} finally { Pop-Location }
if ($code -ne 0) { exit $code }
Write-Host "Review files: $outputPath"
Write-Host 'All labels start as PENDING or UNKNOWN. Fill them after reviewing actual evidence.'
