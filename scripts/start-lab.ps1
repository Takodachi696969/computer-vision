[CmdletBinding()]
param([int]$Port = 8765, [string]$Scene = '')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run scripts/bootstrap.ps1 first.' }
$taskUrl = "http://127.0.0.1:$Port"
try {
    $taskHealth = Invoke-RestMethod "$taskUrl/health" -TimeoutSec 2
    if ($taskHealth.mode -eq 'physics' -and $taskHealth.ok) {
        if ($Scene) { throw 'A server is already running. Stop it before selecting a different scene.' }
        Write-Host "Lab already running: $taskUrl"
        return
    }
    throw 'Port is used by another service.'
} catch {
    # A listener check below distinguishes unavailable service from an occupied port.
    $taskListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($taskListener) { throw "Port $Port is occupied; stop the existing server or choose another -Port." }
}
New-Item -ItemType Directory -Force -Path (Join-Path $taskRoot 'outputs') | Out-Null
$taskArgs = @('-m', 'humaned_lab', 'serve', '--port', "$Port")
if ($Scene) {
    $taskScenePath = (Resolve-Path -LiteralPath $Scene).Path
    $taskArgs += @('--scene', ('"' + $taskScenePath + '"'))
}
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRoot 'outputs\server.stdout.log') -RedirectStandardError (Join-Path $taskRoot 'outputs\server.stderr.log')
$taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRoot "outputs\server-$Port.pid")
for ($taskAttempt=0; $taskAttempt -lt 30; $taskAttempt++) {
    try {
        $taskHealth = Invoke-RestMethod "$taskUrl/health" -TimeoutSec 2
        if ($taskHealth.ok) { Write-Host "Lab running: $taskUrl (PID $($taskProcess.Id))"; return }
    } catch { }
    if ($taskProcess.HasExited) { throw 'Server exited; inspect outputs/server.stderr.log.' }
    Start-Sleep -Milliseconds 500
}
throw 'Server did not become healthy; inspect outputs/server.stderr.log.'

