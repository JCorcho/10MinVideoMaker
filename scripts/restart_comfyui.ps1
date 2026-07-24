param(
    [Parameter(Mandatory = $true)]
    [string]$EasyInstallRoot,
    [Parameter(Mandatory = $true)]
    [string]$ProjectRuntimeRoot,
    [int]$Port = 8188
)

$ErrorActionPreference = "Stop"
$resolvedBase = (Resolve-Path -LiteralPath $EasyInstallRoot).Path
$resolvedRuntime = (Resolve-Path -LiteralPath $ProjectRuntimeRoot).Path
$startupScript = Join-Path $resolvedBase "Start ComfyUI.bat"
$expectedPython = Join-Path $resolvedBase "python_embeded\python.exe"

if (-not (Test-Path -LiteralPath $startupScript -PathType Leaf)) {
    throw "ComfyUI startup script is missing: $startupScript"
}
if (-not (Test-Path -LiteralPath $expectedPython -PathType Leaf)) {
    throw "Embedded Python is missing: $expectedPython"
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    $resolvedExecutable = (Resolve-Path -LiteralPath $process.ExecutablePath).Path
    if (-not $resolvedExecutable.Equals($expectedPython, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Port $Port belongs to an unexpected process: $resolvedExecutable"
    }
    Stop-Process -Id $process.ProcessId -Force
    Start-Sleep -Seconds 2
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $resolvedRuntime "supervisor-restart-$stamp.out.log"
$stderr = Join-Path $resolvedRuntime "supervisor-restart-$stamp.err.log"
$env:PYTHONUTF8 = "1"
Start-Process cmd.exe `
    -ArgumentList "/c", "call `"Start ComfyUI.bat`"" `
    -WorkingDirectory $resolvedBase `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr | Out-Null

$deadline = (Get-Date).AddMinutes(3)
do {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/system_stats" -TimeoutSec 3 | Out-Null
        exit 0
    }
    catch {
        Start-Sleep -Seconds 3
    }
} while ((Get-Date) -lt $deadline)

throw "ComfyUI did not become healthy within three minutes. Logs: $stdout and $stderr"
