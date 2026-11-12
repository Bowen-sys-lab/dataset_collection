param(
    [string]$LocalDirectory = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$lock = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'dependency-lock.json') | ConvertFrom-Json
$dependencyRoot = Join-Path $projectRoot $lock.directory
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows first.' }
if ($LocalDirectory) {
    $sourceRoot = (Resolve-Path -LiteralPath $LocalDirectory -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot '.git'))) {
        throw "Local thermal dependency must be a Git checkout at the pinned revision: $sourceRoot"
    }
    if (Test-Path -LiteralPath $dependencyRoot) {
        throw "Destination already exists: $dependencyRoot. Remove or choose a clean project copy."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dependencyRoot) | Out-Null
    & git clone --config core.autocrlf=true --no-checkout $sourceRoot $dependencyRoot
    if ($LASTEXITCODE -ne 0) { throw 'Local thermal dependency clone failed.' }
    & git -C $dependencyRoot checkout --detach $lock.commit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned revision is unavailable in the local thermal checkout.' }
} elseif (Test-Path -LiteralPath $dependencyRoot) {
    if (-not (Test-Path -LiteralPath (Join-Path $dependencyRoot '.git'))) { throw "Existing dependency directory is not a Git checkout: $dependencyRoot" }
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dependencyRoot) | Out-Null
    # Match the Windows checkout used to produce dependency-lock.json hashes.
    & git clone --config core.autocrlf=true --no-checkout $lock.repository $dependencyRoot
    if ($LASTEXITCODE -ne 0) { throw 'Clone failed; check upstream access and its terms.' }
    & git -C $dependencyRoot checkout --detach $lock.commit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned revision is unavailable.' }
}
$actual = & git -C $dependencyRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actual -ne $lock.commit) { throw 'Existing dependency is not the pinned revision. Preserve it and use a clean directory.' }
foreach ($row in $lock.files) {
    $target = Join-Path $dependencyRoot $row.path
    if ($row.generated_cache) {
        $source = Join-Path $PSScriptRoot ('runtime-cache/' + $row.path)
        if (-not (Test-Path -LiteralPath $target)) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            Copy-Item -LiteralPath $source -Destination $target
        }
    }
    if (-not (Test-Path -LiteralPath $target)) { throw "Dependency file missing: $target" }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw "Dependency hash mismatch: $target" }
}
Write-Host "Thermal dependency verified: $actual"
