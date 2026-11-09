param([string]$Out = 'D:\IsaacSim_Demo\validation\performance\preview-optimization-20261008')
$ErrorActionPreference = 'Stop'
$Project = 'D:\IsaacSim_Demo'
New-Item -ItemType Directory -Path $Out -Force | Out-Null
$Launcher = Join-Path $Project 'scripts\run_isaac_with_clean_env.ps1'
$Preview = Join-Path $Project 'scripts\start_crazyflie_dual_camera_view.py'
$Config = Join-Path $Project 'configs\school_vln_scene.yaml'
# Sequential fresh Kit processes: never compete for the same GPU.
$Jobs = @(
    @{ Name='baseline-headless-160-v2'; Width=160; Height=128; Extra=@('--explicit-render','--headless') },
    @{ Name='live-gui-160-v2'; Width=160; Height=128; Extra=@('--preview-mode','live') },
    @{ Name='live-gui-320-v2'; Width=320; Height=256; Extra=@('--preview-mode','live') },
    @{ Name='live-gui-640-v2'; Width=640; Height=512; Extra=@('--preview-mode','live') }
)
foreach ($Job in $Jobs) {
    $Run = Join-Path $Out $Job.Name
    if (Test-Path (Join-Path $Run 'performance.json')) { continue }
    $ExtraArgs = $Job.Extra
    & $Launcher $Preview --config $Config --width $Job.Width --height $Job.Height --frames 40 --no-save --probe-control --disable-external-viewport --out $Run @ExtraArgs *> "$Run.log"
    Write-Host "$($Job.Name): exit=$LASTEXITCODE"
}
$Probe = Join-Path $Project 'tests\isaac_dual_camera_sync.py'
$Realtime = Join-Path $Out 'realtime-marker-160'
if (-not (Test-Path (Join-Path $Realtime 'summary.json'))) {
    & $Launcher $Probe --config $Config --renderer RealTimePathTracing --frames 8 --width 160 --height 128 --out $Realtime *> "$Realtime.log"
}
Write-Host "realtime marker: exit=$LASTEXITCODE"
$Proof = Join-Path $Realtime 'summary.json'
if ((Test-Path $Proof) -and ((Get-Content $Proof -Raw | ConvertFrom-Json).result -eq 'PASS')) {
    $Run = Join-Path $Out 'realtime-live-gui-160'
    if (-not (Test-Path (Join-Path $Run 'performance.json'))) {
        & $Launcher $Preview --config $Config --renderer RealTimePathTracing --renderer-proof $Proof --preview-mode live --width 160 --height 128 --frames 40 --no-save --probe-control --disable-external-viewport --out $Run *> "$Run.log"
    }
}
$Regression = Join-Path $Out 'offline-marker-640'
if (-not (Test-Path (Join-Path $Regression 'summary.json'))) {
    & $Launcher $Probe --config $Config --renderer PathTracing --frames 8 --width 640 --height 512 --out $Regression *> "$Regression.log"
}
Write-Host "offline regression: exit=$LASTEXITCODE"
