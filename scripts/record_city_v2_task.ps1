param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Task,
    [string]$RunId = '',
    [switch]$CheckOnly,
    [switch]$CollisionUndo = $true
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$table = Get-Content -LiteralPath (Join-Path $projectRoot 'configs\city_200m_v2_task_table.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$entry = $table.tasks | Where-Object { $_.task_id -eq $Task } | Select-Object -First 1
if (-not $entry) { throw "Unknown task '$Task'. See configs\city_200m_v2_task_table.json" }
$taskReference = [string]$entry.file
$legacyRoot = 'D:\IsaacSim_Demo\'
if ($taskReference.StartsWith($legacyRoot, [StringComparison]::OrdinalIgnoreCase)) {
    $taskReference = $taskReference.Substring($legacyRoot.Length)
}
if ([IO.Path]::IsPathRooted($taskReference)) { throw "Task file must be inside the project: $taskReference" }
$taskFile = [IO.Path]::GetFullPath((Join-Path $projectRoot $taskReference))
if (-not $taskFile.StartsWith($projectRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Task file must be inside the project: $taskFile"
}
if (-not (Test-Path -LiteralPath $taskFile)) { throw "Task file missing: $taskFile" }

# Escape non-ASCII characters so Windows PowerShell can decode native stdout with any console code page.
$rawJson = & python -X utf8 -c "import json,sys,yaml;print(json.dumps(yaml.safe_load(open(sys.argv[1],encoding='utf-8')),ensure_ascii=True))" $taskFile
if ($LASTEXITCODE -ne 0 -or -not $rawJson) { throw "Cannot parse task YAML: $taskFile" }
$taskDoc = ConvertFrom-Json -InputObject ([string]$rawJson)
$start = @($taskDoc.start_position)
$yaw = [double]$taskDoc.environment_evidence.initial_yaw_degrees
if ($start.Count -ne 3) { throw "Task start_position must have 3 values: $taskFile" }

if (-not $RunId) {
    # Match the Forest episode identity: <scene>-pilot30-<episode>-<UTC timestamp>.
    do {
        $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssffffffZ')
        $RunId = "city-pilot30-$($entry.episode)-$stamp"
        $candidate = Join-Path $projectRoot "validation\trajectory_runs\$RunId"
    } while (Test-Path -LiteralPath $candidate)
}

$config = Join-Path $projectRoot 'configs\city_200m_v3_scene.yaml'
$thermalRevision = 'city_material_partition_v3'
$outDir = Join-Path $projectRoot "validation\city-v2\previews\$RunId"
$launchDir = Join-Path $projectRoot 'validation\city-v2\launches'
$captureJson = & python -X utf8 (Join-Path $PSScriptRoot 'collection_capture_defaults.py')
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the capture contract.' }
$captureContract = ConvertFrom-Json -InputObject ([string]$captureJson)
$sha256 = [Security.Cryptography.SHA256]::Create()
try {
    $taskSha256 = [BitConverter]::ToString($sha256.ComputeHash([IO.File]::ReadAllBytes($taskFile))).Replace('-', '').ToLowerInvariant()
} finally {
    $sha256.Dispose()
}
$snapshot = [ordered]@{
    schema_version = 1
    run_id = $RunId
    task_id = $entry.task_id
    task_version = $entry.task_version
    target = $entry.target
    task_file = $taskFile
    task_sha256 = $taskSha256
    scene_config = $config
    thermal_revision = $thermalRevision
    start_position = $start
    initial_yaw_degrees = $yaw
    relative_direction = $entry.direction
    instruction = $taskDoc.instruction
    capture_contract = $captureContract
    collision_undo = [bool]$CollisionUndo
    collision_contract = $(if ($CollisionUndo) { 'static_sphere_undo_v2' } else { 'disabled' })
    created_at = (Get-Date).ToString('o')
}
$snapshotPath = Join-Path $launchDir "$RunId.json"

if ($CheckOnly) {
    Write-Host "[city-v2] scene_config=$config thermal_revision=$thermalRevision"
    Write-Host "[city-v2] task=$($entry.task_id) target=$($entry.target)"
    Write-Host "[city-v2] task_version=$($entry.task_version)"
    Write-Host "[city-v2] start=$($start -join ', ') yaw=$yaw direction=$($entry.direction)"
    Write-Host "[city-v2] instruction=$($taskDoc.instruction)"
    Write-Host "[city-v2] run=$RunId snapshot=$snapshotPath"
    Write-Host '[city-v2] raw trajectory=1.0 simulation seconds; offline pair=1.0 seconds; RGB=256x256 IR=256x256; preview RGB=15 fps IR=2 fps'
    Write-Host "[city-v2] collision_undo=$([bool]$CollisionUndo); manual undo=3 simulation seconds; raw attempts retained"
    exit 0
}

New-Item -ItemType Directory -Path $launchDir -Force | Out-Null
[IO.File]::WriteAllText($snapshotPath, ($snapshot | ConvertTo-Json -Depth 5),
    (New-Object System.Text.UTF8Encoding($false)))

$shellPath = (Get-Process -Id $PID).Path
$scriptArgs = @('--config', $config, '--preview-mode', 'live', '--record-trajectory',
    '--run-id', $RunId, '--no-save', '--speed-scale', '1',
    '--horizontal-speed', '3', '--vertical-speed', '1',
    '--trajectory-period', '1.0', '--preview-fps', '15', '--out', $outDir,
    '--live-refresh', 'on-demand', '--thermal-preview-fps', '2',
    '--start-position', [string]$start[0], [string]$start[1], [string]$start[2],
    '--start-yaw-degrees', [string]$yaw,
    '--task-file', $taskFile)
$scriptArgs += $(if ($CollisionUndo) { '--collision-undo' } else { '--no-collision-undo' })
$clean = (Join-Path $PSScriptRoot 'run_isaac_with_clean_env.ps1').Replace("'", "''")
$script = (Join-Path $PSScriptRoot 'start_crazyflie_dual_camera_view.py').Replace("'", "''")
$quoted = ($scriptArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ','
$child = "& '$clean' -ScriptPath '$script' -ScriptArgs @($quoted)"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($child))
Write-Host "[city-v2] recording $RunId ($($entry.task_id)); click Drone Control, T to confirm."
& $shellPath -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Preview exited with $code; see $outDir" }
$metadata = Join-Path $projectRoot "validation\trajectory_runs\$RunId\run_metadata.json"
if (-not (Test-Path -LiteralPath $metadata)) { throw "No run_metadata.json; recording incomplete: $metadata" }
Write-Host "[city-v2] recorded: $metadata"
Write-Host "[city-v2] replay: powershell -File scripts\process.ps1 -Run $RunId"
exit 0
