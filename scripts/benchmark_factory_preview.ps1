param([string]$Out = 'validation/factory-two-target/20261030/benchmark')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$outPath=Join-Path $projectRoot $Out
if(Test-Path -LiteralPath $outPath){throw "Existing benchmark output: $outPath"}
New-Item -ItemType Directory -Path $outPath | Out-Null
$shellPath=(Get-Process -Id $PID).Path
foreach($round in 1..3){
  foreach($mode in @('paired','split')){
    $name="$mode-$round"
    $destination=Join-Path $outPath $name
    $clean=(Join-Path $PSScriptRoot 'run_isaac_with_clean_env.ps1').Replace("'","''")
    $script=(Join-Path $PSScriptRoot 'start_crazyflie_dual_camera_view.py').Replace("'","''")
    $arguments=@('--config',(Join-Path $projectRoot 'configs/model02_v7m_bc_pilot_scene.yaml'),
      '--headless','--preview-mode','live','--live-refresh',$mode,'--preview-fps','15',
      '--thermal-preview-fps','2','--benchmark-seconds','60','--benchmark-warmup','10',
      '--probe-control','--horizontal-speed','3','--vertical-speed','1','--no-save','--out',$destination)
    $literalArgs=($arguments | ForEach-Object { "'"+$_.Replace("'","''")+"'" }) -join ','
    $command="& '$clean' -ScriptPath '$script' -ScriptArgs @($literalArgs)"
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    Write-Host "Benchmark $name started"
    & $shellPath -NoProfile -OutputFormat Text -EncodedCommand $encoded *> (Join-Path $outPath "$name.log")
    if($LASTEXITCODE -ne 0){throw "Benchmark child $name failed; inspect log"}
    $report=Join-Path $destination 'performance.json'
    if(-not(Test-Path -LiteralPath $report)){throw "Missing performance report $name"}
    $data=Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if($data.completion -ne 'completed' -or $data.benchmark_measurement_s -lt 60){throw "Incomplete measurement $name"}
    Write-Host "Benchmark $name completed"
  }
}
