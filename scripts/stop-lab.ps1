[CmdletBinding()]
param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPidFile = Join-Path $taskRoot "outputs\server-$Port.pid"
if (-not (Test-Path -LiteralPath $taskPidFile)) { throw 'No PID recorded; stop the foreground server with Ctrl+C.' }
$taskProcessId = [int](Get-Content -LiteralPath $taskPidFile -Raw).Trim()
$taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$taskProcessId" -ErrorAction SilentlyContinue
if ($taskProcess) {
    $taskExpected = [IO.Path]::GetFullPath((Join-Path $taskRoot '.venv\Scripts\python.exe'))
    if ($taskProcess.ExecutablePath -ne $taskExpected -or $taskProcess.CommandLine -notmatch 'humaned_lab.*serve') {
        throw 'PID belongs to another process; refusing to stop it.'
    }
    Stop-Process -Id $taskProcessId
}
Remove-Item -LiteralPath $taskPidFile
Write-Host "Stopped the recorded lab server on port $Port."

