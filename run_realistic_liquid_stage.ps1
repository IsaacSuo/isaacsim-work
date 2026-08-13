[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("simulate-cache", "render-segment")]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$Job,

    [int]$Start = -1,
    [int]$End = -1,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-f0-9]{32}$")]
    [string]$RunId,

    [string]$ProjectDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSCommandPath

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $temporary = "$Path.$RunId.tmp"
    $json = $Value | ConvertTo-Json -Depth 12
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, $utf8NoBom)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-RelativeJobPath {
    param([string]$RelativePath)
    return Join-Path -Path $Job -ChildPath ($RelativePath -replace '/', '\')
}

$jobPath = Join-Path $Job "job.json"
if (-not (Test-Path -LiteralPath $jobPath -PathType Leaf)) {
    throw "Missing job.json: $jobPath"
}
$jobData = Get-Content -LiteralPath $jobPath -Raw | ConvertFrom-Json
$logs = Get-RelativeJobPath $jobData.paths.logs_dir
New-Item -ItemType Directory -Path $logs -Force | Out-Null

$python = $env:ISAACSIM_PYTHON
if ([string]::IsNullOrWhiteSpace($python)) {
    $python = "Y:\isaacsim\python.bat"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing Isaac Sim Python. Set ISAACSIM_PYTHON or install at: $python"
}

if ($Stage -eq "render-segment") {
    if ($Start -lt 0 -or $End -lt $Start) {
        throw "Invalid render range $Start..$End"
    }
    $script = Join-Path $ProjectDir "render_realistic_liquid_cache.py"
    $name = "render_{0:D6}_{1:D6}_{2}" -f $Start, $End, $RunId
    $scriptArgs = @(
        "--job", $Job,
        "--start", [string]$Start,
        "--end", [string]$End,
        "--run-id", $RunId
    )
}
else {
    $script = Join-Path $ProjectDir "physx_realistic_liquid.py"
    $name = "simulate_cache_$RunId"
    $scriptArgs = @("--video-job", $Job)
}
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
    throw "Missing stage script: $script"
}

$runLog = Join-Path $logs "$name.log"
$monitorCsv = Join-Path $logs "${name}_gpu_samples.csv"
$monitorPid = Join-Path $logs "${name}_gpu_monitor.pid"
$gpuJson = Join-Path $logs "${name}_gpu_peak.json"
$logJson = Join-Path $logs "${name}_log_validation.json"

$monitor = $null
$stageExitCode = 1
try {
    $monitor = Start-Process nvidia-smi `
        -ArgumentList "--query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits --loop=1" `
        -RedirectStandardOutput $monitorCsv `
        -WindowStyle Hidden `
        -PassThru
    $monitor.Id | Set-Content -LiteralPath $monitorPid -Encoding ASCII

    Write-Host "[long-video] run=$RunId stage=$Stage"
    Write-Host "[long-video] logging live output to $runLog"
    & $python $script @scriptArgs 2>&1 | Tee-Object -FilePath $runLog
    $stageExitCode = $LASTEXITCODE
}
finally {
    if ($null -ne $monitor) {
        Stop-Process -Id $monitor.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $monitor.Id -ErrorAction SilentlyContinue
    }
}

$limit = [int]$jobData.resource_budget.hard_vram_limit_mib
$values = @()
if (Test-Path -LiteralPath $monitorCsv -PathType Leaf) {
    $values = @(
        Get-Content -LiteralPath $monitorCsv | ForEach-Object {
            if ($_ -match ',\s*(\d+)\s*,') {
                [int]$Matches[1]
            }
        }
    )
}
$peak = ($values | Measure-Object -Maximum).Maximum
$gpuPassed = ($values.Count -gt 0 -and $peak -le $limit)
$gpuReport = [ordered]@{
    schema_version = 1
    run_id = $RunId
    peak_memory_mib = $peak
    limit_mib = $limit
    passed = $gpuPassed
    sample_count = $values.Count
}
Write-AtomicJson -Value $gpuReport -Path $gpuJson

$lines = @()
if (Test-Path -LiteralPath $runLog -PathType Leaf) {
    $lines = @(Get-Content -LiteralPath $runLog)
}
$needles = @(
    "Particle system contact buffer overflow",
    "CUDA out of memory",
    "CUDA OOM",
    "non-finite",
    "Fixed particle topology changed",
    "Timed out waiting for capture",
    "Capture did not produce a non-empty file",
    "Traceback (most recent call last)",
    "RuntimeError:",
    "Boost.Python.ArgumentError"
)
$fatal = @($lines | Select-String -SimpleMatch -Pattern $needles)
$errors = @($lines | Select-String -SimpleMatch "[Error]")
$logReport = [ordered]@{
    schema_version = 1
    run_id = $RunId
    passed = ($fatal.Count -eq 0)
    fatal_count = $fatal.Count
    fatal_lines = @($fatal | ForEach-Object { $_.Line } | Select-Object -First 100)
    error_count = $errors.Count
    error_lines = @($errors | ForEach-Object { $_.Line } | Select-Object -First 100)
}
Write-AtomicJson -Value $logReport -Path $logJson

if ($stageExitCode -ne 0) {
    Write-Error "Stage exited with $stageExitCode. See $runLog"
    exit $stageExitCode
}
if (-not $logReport.passed) {
    Write-Error "Stage log contains fatal errors. See $logJson"
    exit 5
}
if (-not $gpuPassed) {
    Write-Error "Stage exceeded VRAM budget or produced no samples. See $gpuJson"
    exit 6
}

if ($Stage -eq "render-segment") {
    $segmentStem = "segment_{0:D6}_{1:D6}" -f $Start, $End
    $segmentsDir = Get-RelativeJobPath $jobData.paths.render_segments_dir
    $manifest = Join-Path $segmentsDir "$segmentStem.jsonl"
    $completePath = Join-Path $segmentsDir "${segmentStem}_complete.json"
    $acceptedPath = Join-Path $segmentsDir "${segmentStem}_accepted.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf) -or
        -not (Test-Path -LiteralPath $completePath -PathType Leaf)) {
        throw "Renderer did not write segment manifest/completion"
    }
    $complete = Get-Content -LiteralPath $completePath -Raw | ConvertFrom-Json
    if (-not $complete.valid -or [string]$complete.run_id -ne $RunId) {
        throw "Renderer completion marker is invalid or belongs to another run"
    }
    $accepted = [ordered]@{
        schema_version = 1
        valid = $true
        run_id = $RunId
        start = $Start
        end = $End
        take_id = [string]$jobData.take_id
        config_hash = [string]$jobData.config_hash
        simulation_provenance_hash = [string]$jobData.simulation_provenance_hash
        render_provenance_hash = [string]$complete.render_provenance_hash
        manifest_sha256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
        completion_sha256 = (Get-FileHash -LiteralPath $completePath -Algorithm SHA256).Hash.ToLowerInvariant()
        gpu_report = [IO.Path]::GetFileName($gpuJson)
        gpu_report_sha256 = (Get-FileHash -LiteralPath $gpuJson -Algorithm SHA256).Hash.ToLowerInvariant()
        log_report = [IO.Path]::GetFileName($logJson)
        log_report_sha256 = (Get-FileHash -LiteralPath $logJson -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    Write-AtomicJson -Value $accepted -Path $acceptedPath
}
else {
    $takeDir = Get-RelativeJobPath $jobData.paths.take_dir
    $completionPath = Get-RelativeJobPath $jobData.paths.simulation_complete
    $manifestPath = Get-RelativeJobPath $jobData.paths.simulation_manifest
    $acceptedPath = Join-Path $takeDir "simulation_accepted.json"
    if (-not (Test-Path -LiteralPath $completionPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Simulation did not write manifest/completion"
    }
    $completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
    if (-not $completion.valid -or
        [string]$completion.take_id -ne [string]$jobData.take_id -or
        [string]$completion.config_hash -ne [string]$jobData.config_hash -or
        [string]$completion.simulation_provenance_hash -ne [string]$jobData.simulation_provenance_hash -or
        [int]$completion.frame_count -ne [int]$jobData.video.output_frames) {
        throw "Simulation completion marker failed identity/count validation"
    }
    $accepted = [ordered]@{
        schema_version = 1
        valid = $true
        run_id = $RunId
        take_id = [string]$jobData.take_id
        config_hash = [string]$jobData.config_hash
        simulation_provenance_hash = [string]$jobData.simulation_provenance_hash
        manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        completion_sha256 = (Get-FileHash -LiteralPath $completionPath -Algorithm SHA256).Hash.ToLowerInvariant()
        gpu_report = [IO.Path]::GetFileName($gpuJson)
        gpu_report_sha256 = (Get-FileHash -LiteralPath $gpuJson -Algorithm SHA256).Hash.ToLowerInvariant()
        log_report = [IO.Path]::GetFileName($logJson)
        log_report_sha256 = (Get-FileHash -LiteralPath $logJson -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    Write-AtomicJson -Value $accepted -Path $acceptedPath
}

Write-Host "[long-video] accepted run $RunId"
exit 0
