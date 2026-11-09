param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ScriptPath,

    [Parameter(ValueFromRemainingArguments = $true, Position = 1)]
    [string[]]$ScriptArgs
)

# Keep Isaac/ USD DLL discovery inside this child process.  In particular, do
# not let WindowsImportWrapper inspect the full user PATH (which may contain
# WindowsApps or long Codex paths).  Nothing here changes system variables.
$IsaacRoot = 'D:\isaacsim'
$ProjectRoot = 'D:\IsaacSim_Demo'
$UsdRoot = Join-Path $IsaacRoot 'extscache\omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312'

$env:PXR_USD_WINDOWS_DLL_PATH = @(
    (Join-Path $IsaacRoot 'kit'),
    (Join-Path $IsaacRoot 'kit\bin'),
    (Join-Path $IsaacRoot 'kit\python'),
    (Join-Path $UsdRoot 'bin')
) -join ';'
$env:PYTHONPATH = @(
    $ProjectRoot,
    (Join-Path $ProjectRoot 'third_party\isaac-thermal-camera-simulation\src')
) -join ';'
# Replicator imports Warp during Kit startup.  Use a short, per-project cache
# path so parallel/previous runs cannot collide in the user profile cache.
$warpCache = Join-Path $ProjectRoot 'validation\isaac_cache\warp'
New-Item -ItemType Directory -Path $warpCache -Force | Out-Null
$env:WARP_CACHE_PATH = $warpCache
$env:OMNI_WARP_CACHE_PATH = $warpCache
$env:PATH = @(
    (Join-Path $IsaacRoot 'kit'),
    (Join-Path $IsaacRoot 'kit\bin'),
    (Join-Path $IsaacRoot 'kit\python'),
    'C:\Windows\System32',
    'C:\Windows',
    'C:\Windows\System32\Wbem'
) -join ';'

$resolvedScript = (Resolve-Path -LiteralPath $ScriptPath).Path
$pythonBat = Join-Path $IsaacRoot 'python.bat'
if (-not (Test-Path -LiteralPath $pythonBat)) {
    throw "Isaac Python launcher not found: $pythonBat"
}

Write-Host "PXR_USD_WINDOWS_DLL_PATH=$env:PXR_USD_WINDOWS_DLL_PATH"
Write-Host "PYTHONPATH=$env:PYTHONPATH"
& $pythonBat -B $resolvedScript @ScriptArgs
exit $LASTEXITCODE
