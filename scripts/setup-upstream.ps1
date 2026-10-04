[CmdletBinding()]
param([string]$SourcePath = '')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskRevision = 'b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4'
if (-not $SourcePath) {
    $SourcePath = Join-Path $taskRoot 'vendor\PAROL6-python-API'
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        & git clone https://github.com/PCrnjak/PAROL6-python-API.git $SourcePath
        if ($LASTEXITCODE -ne 0) { throw 'Upstream clone failed.' }
        & git -C $SourcePath checkout $taskRevision
        if ($LASTEXITCODE -ne 0) { throw 'Pinned checkout failed.' }
    }
}
$taskSource = (Resolve-Path -LiteralPath $SourcePath).Path
$taskActual = (& git -C $taskSource rev-parse HEAD).Trim()
if ($taskActual -ne $taskRevision) { throw "Expected upstream $taskRevision, found $taskActual. Use a separate pinned checkout." }
$taskDirty = & git -C $taskSource status --porcelain
if ($taskDirty) { throw 'Upstream checkout has local edits; use a clean pinned checkout.' }
& uv venv --python 3.12 --allow-existing .upstream-venv
if ($LASTEXITCODE -ne 0) { throw 'Upstream virtual environment creation failed.' }
$taskPython = Join-Path $taskRoot '.upstream-venv\Scripts\python.exe'
# TOPPRA has no Windows wheel. Install Microsoft C++ Build Tools (Desktop C++ workload)
# first, or use Linux/WSL for this optional upstream controller environment.
& uv pip install --python $taskPython --constraint (Join-Path $taskRoot 'requirements\upstream-constraints.txt') -e $taskSource -e $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'Upstream install failed. Check MSVC Build Tools / TOPPRA; see docs/tutorial.md.' }
& $taskPython -m humaned_lab upstream-smoke
if ($LASTEXITCODE -ne 0) { throw 'Upstream mock controller test failed.' }
