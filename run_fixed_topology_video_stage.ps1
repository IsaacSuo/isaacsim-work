[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("simulate-cache", "render-segment")]
    [string]$Stage,
    [Parameter(Mandatory = $true)][string]$Job,
    [int]$Start = -1,
    [int]$End = -1,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-f0-9]{32}$")]
    [string]$RunId
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSCommandPath
$jobPath = Join-Path $Job "job.json"
if (-not (Test-Path -LiteralPath $jobPath -PathType Leaf)) { throw "Missing job.json: $jobPath" }
$jobData = Get-Content -LiteralPath $jobPath -Raw | ConvertFrom-Json
if ([string]$jobData.job_type -ne "fixed_topology_mesh_video") { throw "Unsupported job type" }
$logs = Join-Path $Job ([string]$jobData.paths.logs_dir -replace '/', '\')
New-Item -ItemType Directory -Path $logs -Force | Out-Null

$python = $env:ISAACSIM_PYTHON
if ([string]::IsNullOrWhiteSpace($python)) { $python = "Y:\isaacsim\python.bat" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing Isaac Sim Python: $python" }

if ($Stage -eq "simulate-cache") {
    $script = Join-Path $project ([string]$jobData.simulation.script)
    $name = "simulate_cache_$RunId"
    $scriptArgs = @("--video-job", $Job)
}
else {
    if ($Start -lt 0 -or $End -lt $Start) { throw "Invalid render range $Start..$End" }
    $script = Join-Path $project "render_fixed_topology_video.py"
    $name = "render_{0:D6}_{1:D6}_{2}" -f $Start, $End, $RunId
    $scriptArgs = @("--job", $Job, "--start", [string]$Start, "--end", [string]$End, "--run-id", $RunId)
}
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) { throw "Missing stage script: $script" }

$runLog = Join-Path $logs "$name.log"
$monitorCsv = Join-Path $logs "${name}_gpu_samples.csv"
$gpuJson = Join-Path $logs "${name}_gpu_peak.json"
$logJson = Join-Path $logs "${name}_log_validation.json"
$monitor = $null
$stageExitCode = 1
try {
    $monitor = Start-Process nvidia-smi `
        -ArgumentList "--query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits --loop=1" `
        -RedirectStandardOutput $monitorCsv -WindowStyle Hidden -PassThru
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $python $script @scriptArgs 2>&1 | Tee-Object -FilePath $runLog
        $stageExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}
finally {
    if ($null -ne $monitor) {
        Stop-Process -Id $monitor.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $monitor.Id -ErrorAction SilentlyContinue
    }
}

function Write-Json([object]$Value, [string]$Path) {
    $temporary = "$Path.$RunId.tmp"
    $json = $Value | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$values = @()
if (Test-Path -LiteralPath $monitorCsv) {
    $values = @(Get-Content -LiteralPath $monitorCsv | ForEach-Object { if ($_ -match ',\s*(\d+)\s*,') { [int]$Matches[1] } })
}
$peak = ($values | Measure-Object -Maximum).Maximum
$limit = [int]$jobData.resource_budget.hard_vram_limit_mib
$gpuPassed = ($values.Count -gt 0 -and $peak -le $limit)
Write-Json ([ordered]@{schema_version=1;run_id=$RunId;peak_memory_mib=$peak;limit_mib=$limit;passed=$gpuPassed;sample_count=$values.Count}) $gpuJson

$lines = if (Test-Path -LiteralPath $runLog) { @(Get-Content -LiteralPath $runLog) } else { @() }
$needles = @("CUDA out of memory", "CUDA OOM", "non-finite", "Traceback (most recent call last)", "RuntimeError:", "Boost.Python.ArgumentError")
$fatal = @($lines | Select-String -SimpleMatch -Pattern $needles)
$logReport = [ordered]@{schema_version=1;run_id=$RunId;passed=($fatal.Count -eq 0);fatal_count=$fatal.Count;fatal_lines=@($fatal | ForEach-Object {$_.Line} | Select-Object -First 100)}
Write-Json $logReport $logJson
if ($stageExitCode -ne 0) { throw "Stage exited with $stageExitCode. See $runLog" }
if (-not $logReport.passed) { throw "Stage log contains fatal errors. See $logJson" }
if (-not $gpuPassed) { throw "Stage exceeded VRAM budget or produced no samples. See $gpuJson" }

if ($Stage -eq "simulate-cache") {
    $takeDir = Join-Path $Job ([string]$jobData.paths.take_dir -replace '/', '\')
    $completion = Join-Path $Job ([string]$jobData.paths.simulation_complete -replace '/', '\')
    $manifest = Join-Path $Job ([string]$jobData.paths.simulation_manifest -replace '/', '\')
    $acceptedPath = Join-Path $takeDir "simulation_accepted.json"
    if (-not (Test-Path $completion) -or -not (Test-Path $manifest)) { throw "Simulation did not write completion and manifest" }
    $completeData = Get-Content $completion -Raw | ConvertFrom-Json
    if (-not $completeData.valid -or [int]$completeData.frame_count -ne [int]$jobData.video.output_frames) { throw "Invalid simulation completion" }
    $accepted = [ordered]@{schema_version=1;valid=$true;run_id=$RunId;take_id=[string]$jobData.take_id;config_hash=[string]$jobData.config_hash;simulation_provenance_hash=[string]$jobData.simulation_provenance_hash;manifest_sha256=(Get-FileHash $manifest -Algorithm SHA256).Hash.ToLowerInvariant();completion_sha256=(Get-FileHash $completion -Algorithm SHA256).Hash.ToLowerInvariant()}
}
else {
    $segments = Join-Path $Job ([string]$jobData.paths.render_segments_dir -replace '/', '\')
    $stem = "segment_{0:D6}_{1:D6}" -f $Start, $End
    $manifest = Join-Path $segments "$stem.jsonl"
    $completion = Join-Path $segments "${stem}_complete.json"
    $acceptedPath = Join-Path $segments "${stem}_accepted.json"
    if (-not (Test-Path $completion) -or -not (Test-Path $manifest)) { throw "Renderer did not write completion and manifest" }
    $completeData = Get-Content $completion -Raw | ConvertFrom-Json
    if (-not $completeData.valid -or [string]$completeData.run_id -ne $RunId) { throw "Invalid render completion" }
    $accepted = [ordered]@{schema_version=1;valid=$true;run_id=$RunId;start=$Start;end=$End;take_id=[string]$jobData.take_id;config_hash=[string]$jobData.config_hash;simulation_provenance_hash=[string]$jobData.simulation_provenance_hash;render_provenance_hash=[string]$completeData.render_provenance_hash;manifest_sha256=(Get-FileHash $manifest -Algorithm SHA256).Hash.ToLowerInvariant();completion_sha256=(Get-FileHash $completion -Algorithm SHA256).Hash.ToLowerInvariant()}
}
Write-Json $accepted $acceptedPath
Write-Host "[fixed-video] accepted run $RunId"
