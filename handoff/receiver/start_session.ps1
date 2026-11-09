# Dot-source this file so PATH changes apply only to the current PowerShell session.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($projectRoot -ine 'D:\IsaacSim_Demo') { throw 'This delivery requires D:\IsaacSim_Demo. See the Word setup guide.' }
if (-not (Test-Path -LiteralPath 'D:\isaacsim\python.bat')) { throw 'Install the pinned Isaac Sim build at D:\isaacsim first.' }
$env:PATH = (Join-Path $PSScriptRoot 'bin') + ';' + $env:PATH
Set-Location -LiteralPath $projectRoot
Write-Host 'Session ready. python resolves to the Isaac Python launcher in this PowerShell session.'
