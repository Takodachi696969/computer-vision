<#!
.SYNOPSIS
Install the locked Python 3.12 lab without WSL or a C/C++ compiler.
#>
[CmdletBinding()]
param([switch]$Camera, [switch]$CoreOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskUvVersion = (Get-Content -LiteralPath (Join-Path $taskRoot '.uv-version') -Raw).Trim()
$taskUv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $taskUv) {
    throw 'Install uv first: winget install --id astral-sh.uv -e. Then reopen PowerShell.'
}
$taskActualVersion = (& uv --version).Split(' ')[1]
if ($taskActualVersion -ne $taskUvVersion) {
    Write-Warning "Validated uv version is $taskUvVersion; installed version is $taskActualVersion."
}
& uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
$taskArgs = @('sync', '--frozen', '--python', '3.12', '--group', 'dev')
if (-not $CoreOnly) { $taskArgs += @('--extra', 'train', '--extra', 'hub') }
if ($Camera) { $taskArgs += @('--extra', 'camera') }
& uv @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\humaned-lab.exe' doctor
if ($LASTEXITCODE -ne 0) { throw 'Lab self-check failed.' }
Write-Host 'Ready. Start: .\.venv\Scripts\humaned-lab.exe serve'
Write-Host 'Dashboard: http://127.0.0.1:8765'
