param(
    [switch]$InProcess = $true,
    [string]$ApiBaseUrl = "http://localhost:8000",
    [string]$OutputPath = "artifacts/bench/smoke_report_ps1.json",
    [string]$WhenLocal = "2025-09-02T10:30",
    [ValidateRange(1, 5)]
    [int]$Priority = 2,
    [ValidateSet("depart_after", "arrive_before")]
    [string]$Mode = "depart_after",
    [string]$StartLocation,
    [string]$EndLocation
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SmokeScript = Join-Path $PSScriptRoot "smoke_plan_endpoints.py"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

if (-not (Test-Path $SmokeScript)) {
    throw "Smoke script not found at $SmokeScript"
}

$ArgsList = @(
    $SmokeScript,
    "--output", $OutputPath,
    "--when-local", $WhenLocal,
    "--priority", $Priority.ToString(),
    "--mode", $Mode
)

if ($InProcess) {
    $ArgsList += "--in-process"
}
else {
    $ArgsList += @("--api-base-url", $ApiBaseUrl)
}

if ($StartLocation) {
    $ArgsList += @("--start-location", $StartLocation)
}

if ($EndLocation) {
    $ArgsList += @("--end-location", $EndLocation)
}

Write-Host "Running backend smoke check..."
Write-Host "Mode: $(if ($InProcess) { 'in-process' } else { "live ($ApiBaseUrl)" })"
Write-Host "Output: $OutputPath"

& $PythonExe @ArgsList
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    throw "Smoke check failed with exit code $ExitCode"
}

Write-Host "Smoke check completed successfully."
