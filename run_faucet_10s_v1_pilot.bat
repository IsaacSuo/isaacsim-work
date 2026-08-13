@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "PROJECT=Y:\isaacsim_work"
set "JOB=%PROJECT%\output\long_video\faucet_10s_v1"
set "PIPELINE=%PROJECT%\liquid_video_pipeline.py"
set "WSL_PIPELINE=/mnt/y/isaacsim_work/liquid_video_pipeline.py"
set "WSL_JOB=/mnt/y/isaacsim_work/output/long_video/faucet_10s_v1"
set "TAKE=%JOB%\takes\take_551453b88fcf4094823f6d47e8d9d371"
set "SEGMENTS=%JOB%\renders\path_1280x720_30fps_32spp\segments"

if not exist "%PIPELINE%" (
  echo [pilot-preflight] Missing pipeline: %PIPELINE%
  exit /b 2
)
if not exist "%JOB%\job.json" (
  echo [pilot-preflight] Missing job.json
  exit /b 3
)
if not exist "%TAKE%\simulation_complete.json" (
  echo [pilot-preflight] Missing simulation_complete.json
  exit /b 4
)
if not exist "%TAKE%\simulation_accepted.json" (
  echo [pilot-preflight] Missing simulation_accepted.json
  exit /b 5
)
if exist "%SEGMENTS%\segment_000000_000049.jsonl" (
  echo [pilot-preflight] First segment already exists; use pipeline resume/status instead
  exit /b 6
)

powershell -NoProfile -Command ^
  "$j = Get-Content -LiteralPath '%JOB%\job.json' -Raw | ConvertFrom-Json;" ^
  "$ok = $j.video.output_frames -eq 300 -and $j.video.output_fps -eq 30 -and $j.video.width -eq 1280 -and $j.video.height -eq 720 -and $j.render.renderer -eq 'PathTracing' -and $j.render.path_spp -eq 32 -and $j.render.segment_frames -eq 50 -and $j.render.camera -eq 'stream-front';" ^
  "if (-not $ok) { Write-Error 'Pilot profile mismatch'; exit 7 };" ^
  "$s = Get-Content -LiteralPath '%TAKE%\simulation_complete.json' -Raw | ConvertFrom-Json;" ^
  "if (-not $s.valid -or $s.frame_count -ne 300) { Write-Error 'Simulation completion is invalid'; exit 8 };" ^
  "Write-Host '[pilot-preflight] render frames=0..49; encode frames=0..29'"
if errorlevel 1 exit /b %ERRORLEVEL%

if /I "%~1"=="--preflight-only" (
  echo [pilot-preflight] checks passed; pilot was not started
  exit /b 0
)

echo [pilot] starting canonical first segment and one-second encode
wsl.exe python3 "%WSL_PIPELINE%" render-pilot --job "%WSL_JOB%"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo [pilot] failed with exit code %RESULT%
  exit /b %RESULT%
)

echo [pilot] completed and encoded
exit /b 0
