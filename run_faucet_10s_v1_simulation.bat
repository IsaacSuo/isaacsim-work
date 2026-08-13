@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "PROJECT=Y:\isaacsim_work"
set "JOB=%PROJECT%\output\long_video\faucet_10s_v1"
set "STAGE_RUNNER=%PROJECT%\run_realistic_liquid_long_video.bat"

if not exist "%STAGE_RUNNER%" (
  echo [preflight] Missing stage runner: %STAGE_RUNNER%
  exit /b 2
)
if not exist "%JOB%\job.json" (
  echo [preflight] Missing job: %JOB%\job.json
  exit /b 3
)

for /d %%D in ("%JOB%\takes\take_*") do (
  if exist "%%~fD\simulation_complete.json" (
    echo [preflight] Simulation already completed: %%~fD
    exit /b 4
  )
  if exist "%%~fD\simulation_manifest.jsonl" (
    echo [preflight] Partial take exists and cannot resume: %%~fD
    exit /b 5
  )
  if exist "%%~fD\render_template.usda" (
    echo [preflight] Partial take template exists and cannot resume: %%~fD
    exit /b 6
  )
)

powershell -NoProfile -Command ^
  "$j = Get-Content -LiteralPath '%JOB%\job.json' -Raw | ConvertFrom-Json;" ^
  "$ok = $j.video.duration_seconds -eq 10 -and $j.video.output_fps -eq 30 -and $j.video.output_frames -eq 300 -and $j.video.width -eq 1280 -and $j.video.height -eq 720 -and $j.physics.physics_fps -eq 60 -and $j.physics.capture_stride -eq 2 -and $j.physics.source_mode -eq 'emitter' -and $j.physics.emitter_recycle -eq $true -and $j.render.renderer -eq 'PathTracing' -and $j.render.path_spp -eq 32 -and $j.render.camera -eq 'stream-front';" ^
  "if (-not $ok) { Write-Error 'Locked production profile mismatch'; exit 7 };" ^
  "Write-Host ('[preflight] profile=10s, 60Hz, 30fps, 300 frames, 1280x720, 32spp, stream-front')"
if errorlevel 1 exit /b %ERRORLEVEL%

for /f "delims=" %%F in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive -Name Y).Free / 1GB)"') do set "FREE_GIB=%%F"
if not defined FREE_GIB (
  echo [preflight] Could not determine free space on Y:
  exit /b 8
)
if %FREE_GIB% LSS 100 (
  echo [preflight] Insufficient free space: %FREE_GIB% GiB available, 100 GiB required
  exit /b 9
)

for /f "delims=" %%I in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString([char]78)"') do set "RUN_ID=%%I"
if not defined RUN_ID (
  echo [preflight] Could not create run ID
  exit /b 10
)

echo [preflight] job=%JOB%
echo [preflight] free_space=%FREE_GIB% GiB
echo [preflight] run_id=%RUN_ID%
if /I "%~1"=="--preflight-only" (
  echo [preflight] checks passed; simulation was not started
  exit /b 0
)
echo [preflight] starting one non-resumable physical cache take

call "%STAGE_RUNNER%" simulate-cache "%JOB%" "%RUN_ID%"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo [production] Simulation failed with exit code %RESULT%
  exit /b %RESULT%
)

echo [production] Simulation cache take accepted
exit /b 0
