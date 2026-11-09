$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot 'scripts\invoke_logged_powershell.ps1'

Describe 'invoke_logged_powershell' {
    It 'captures plain child stderr without PowerShell XML stream decoding' {
        $powerShell = Join-Path $env:windir 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $testRoot = Join-Path $env:TEMP ('isaac-process-child-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $testRoot | Out-Null
        try {
            $childCode = "[Console]::Out.WriteLine('stdout-marker'); [Console]::Error.WriteLine('stderr-marker'); exit 7"
            $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCode))
            $stdout = Join-Path $testRoot 'stdout.log'
            $stderr = Join-Path $testRoot 'stderr.log'

            $exitCode = & $launcher -ShellPath $powerShell -EncodedCommand $encoded `
                -StdoutPath $stdout -StderrPath $stderr

            $exitCode | Should Be 7
            (Get-Content -LiteralPath $stdout -Raw) | Should Match 'stdout-marker'
            (Get-Content -LiteralPath $stderr -Raw) | Should Match 'stderr-marker'
        } finally {
            Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
