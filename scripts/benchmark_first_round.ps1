param([string]$Out = 'D:\IsaacSim_Demo\validation\performance\first-round-final-20261009')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $Out -Force | Out-Null
$Launcher = 'D:\IsaacSim_Demo\scripts\run_isaac_with_clean_env.ps1'
$Probe = 'D:\IsaacSim_Demo\tests\isaac_live_driving_probe.py'
$Failed = $false
foreach ($Name in @('base','mapping','motion','rgb-only','historical-marker')) {
    [string[]]$Extra = @()
    $Case = 'B'
    if ($Name -eq 'mapping') { $Extra += '--enable-material-cache' }
    if ($Name -eq 'motion') { $Extra += '--compact-motion' }
    if ($Name -eq 'rgb-only') { $Case = 'C' }
    if ($Name -eq 'historical-marker') { $Extra += @('--marker','--compact-motion') }
    $Run = Join-Path $Out $Name
    if (-not (Test-Path $Run)) {
        & $Launcher $Probe --case $Case --frames 100 --preview-hz 0 --skip-preconversion --out $Run @Extra *> "$Run.log"
    }
    $Summary = Join-Path $Run 'summary.json'
    if (-not (Test-Path $Summary)) { $Failed = $true; continue }
    $Result = Get-Content $Summary -Raw | ConvertFrom-Json
    if ($Result.result -ne 'MEASURED') { $Failed = $true }
    Write-Host "$Name result=$($Result.result) fps=$($Result.processing_fps)"
}
if ($Failed) { exit 1 }
