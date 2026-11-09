param(
    [Parameter(Mandatory = $true)]
    [string]$ShellPath,

    [Parameter(Mandatory = $true)]
    [string]$EncodedCommand,

    [Parameter(Mandatory = $true)]
    [string]$StdoutPath,

    [Parameter(Mandatory = $true)]
    [string]$StderrPath
)

$ErrorActionPreference = 'Stop'
$resolvedShell = (Resolve-Path -LiteralPath $ShellPath).Path
# Run the child through cmd.exe so PowerShell never parses the child's stderr
# stream as CLIXML, and Start-Process never copies an environment block that may
# contain case-duplicate keys (NO_PROXY / no_proxy). cmd passes the raw block.
$commandLine = '""{0}" -NoProfile -ExecutionPolicy Bypass -EncodedCommand {1} > "{2}" 2> "{3}""' -f `
    $resolvedShell, $EncodedCommand, $StdoutPath, $StderrPath
& cmd.exe /d /c $commandLine
Write-Output ([int]$LASTEXITCODE)
