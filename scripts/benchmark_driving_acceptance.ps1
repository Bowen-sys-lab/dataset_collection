param([string]$Out = 'D:\IsaacSim_Demo\validation\performance\driving-acceptance-20261008',
      [switch]$FinalRegression, [switch]$SharedProduct)
$ErrorActionPreference = 'Stop'
$Launcher = 'D:\IsaacSim_Demo\scripts\run_isaac_with_clean_env.ps1'
$Probe = 'D:\IsaacSim_Demo\tests\isaac_live_driving_probe.py'
New-Item -ItemType Directory -Path $Out -Force | Out-Null
if ($FinalRegression) {
    foreach ($Renderer in @('PathTracing','RealTimePathTracing')) {
        foreach ($Marker in @($false,$true)) {
            $Prefix = if ($SharedProduct) { 'shared-' } else { '' }
            $Name = $Prefix + $(if ($Marker) { "continuous-B-$Renderer" } else { "optimized-B-$Renderer" })
            $Run = Join-Path $Out $Name
            if (-not (Test-Path (Join-Path $Run 'summary.json'))) {
                [string[]]$Extra = @()
                if ($Marker) { $Extra += @('--marker','--marker-motion','continuous') }
                if ($SharedProduct) { $Extra += @('--shared-render-product') }
                & $Launcher $Probe --case B --frames 100 --renderer $Renderer --skip-preconversion --out $Run @Extra *> "$Run.log"
                Write-Host "$Run exit=$LASTEXITCODE"
            }
        }
    }
    $Run = Join-Path $Out 'offline-marker-640-regression'
    if (-not (Test-Path (Join-Path $Run 'summary.json'))) {
        & $Launcher 'D:\IsaacSim_Demo\tests\isaac_dual_camera_sync.py' --frames 8 --width 640 --height 512 --out $Run *> "$Run.log"
        Write-Host "$Run exit=$LASTEXITCODE"
    }
    return
}
foreach ($Case in @('A','B','C','D')) {
    $Run = Join-Path $Out "baseline-$Case"
    if (-not (Test-Path (Join-Path $Run 'summary.json'))) {
        & $Launcher $Probe --case $Case --frames 100 --skip-preconversion --out $Run *> "$Run.log"
        Write-Host "$Run exit=$LASTEXITCODE"
    }
}
foreach ($Case in @('A','B')) {
    $Run = Join-Path $Out "marker-$Case"
    if (-not (Test-Path (Join-Path $Run 'summary.json'))) {
        & $Launcher $Probe --case $Case --frames 40 --marker --skip-preconversion --out $Run *> "$Run.log"
        Write-Host "$Run exit=$LASTEXITCODE"
    }
}
