@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "OMNI_KIT_ACCEPT_EULA=YES"
set "PYTHON=Y:\isaacsim\python.bat"
set "SCRIPT=Y:\isaacsim_work\physx_realistic_liquid.py"
set "ROOT=Y:\isaacsim_work\output\real_jet_validation_v1"
set "STAGE=%~1"
if not defined STAGE set "STAGE=smoke"

if not exist "%PYTHON%" (
  echo [validation] Missing Isaac Sim Python: %PYTHON%
  exit /b 2
)
if not exist "%SCRIPT%" (
  echo [validation] Missing scene script: %SCRIPT%
  exit /b 2
)
if not exist "%ROOT%" mkdir "%ROOT%"

if /I "%STAGE%"=="plan" goto plan
if /I "%STAGE%"=="smoke" goto smoke
if /I "%STAGE%"=="ab" goto ab
if /I "%STAGE%"=="regression" goto regression
if /I "%STAGE%"=="path-preview" goto path_preview
if /I "%STAGE%"=="path-final" goto path_final
if /I "%STAGE%"=="video-baseline" goto video_baseline
echo Usage: %~nx0 ^<plan^|smoke^|ab^|regression^|path-preview^|path-final^|video-baseline^>
exit /b 2

:plan
call :run plan_default --source-mode emitter --emitter-plan-only
exit /b %errorlevel%

:smoke
call :run coaxial_smoke --source-mode emitter --spacing 0.003 --emitter-nozzle-diameter 0.028 --frames 12 --capture-every 6 --emitter-preroll-frames 12 --emitter-reserve-frames 6 --emitter-ramp-frames 18 --emitter-initial-pool-layers 1 --emitter-initial-pool-span-x 0.20 --emitter-initial-pool-span-z 0.10 --width 640 --height 360 --renderer RaytracedLighting --diagnostic-material --camera-preset stream-front --isosurface-settle-updates 4
exit /b %errorlevel%

:ab
call :run coaxial_physical_b --source-mode emitter --frames 60 --capture-every 10 --emitter-preroll-frames 18 --emitter-reserve-frames 6 --width 640 --height 360 --renderer RaytracedLighting --diagnostic-material --camera-preset stream-front --isosurface-settle-updates 6
exit /b %errorlevel%

:regression
call :run regression_block --source-mode block --frames 3 --capture-every 2 --width 320 --height 180 --renderer RaytracedLighting --diagnostic-material --isosurface-settle-updates 3
if errorlevel 1 exit /b %errorlevel%
call :run regression_stream --source-mode stream --frames 3 --capture-every 2 --reservoir-settle-frames 6 --width 320 --height 180 --renderer RaytracedLighting --diagnostic-material --isosurface-settle-updates 3
exit /b %errorlevel%

:path_preview
call :run path_preview --source-mode emitter --frames 24 --capture-every 12 --emitter-preroll-frames 18 --emitter-reserve-frames 6 --width 1280 --height 720 --renderer PathTracing --path-spp 32 --camera-preset stream-front --isosurface-settle-updates 12
exit /b %errorlevel%

:path_final
call :run path_final --source-mode emitter --frames 60 --capture-every 30 --emitter-preroll-frames 18 --emitter-reserve-frames 6 --width 1280 --height 720 --renderer PathTracing --path-spp 64 --camera-preset stream-front --isosurface-settle-updates 16
exit /b %errorlevel%

:video_baseline
call :run video_buffer_baseline --source-mode emitter --frames 60 --capture-every 30 --emitter-preroll-frames 18 --emitter-reserve-frames 6 --width 1280 --height 720 --renderer PathTracing --path-spp 32 --camera-preset stream-front --isosurface-settle-updates 16 --gpu-max-particle-contacts 1500000
exit /b %errorlevel%

:run
set "NAME=%~1"
set "ARGS="
:collect_args
shift
if "%~1"=="" goto args_ready
set "ARGS=!ARGS! %1"
goto collect_args
:args_ready
set "OUT=%ROOT%\%NAME%"
if exist "%OUT%" (
  echo [validation] Refusing to overwrite %OUT%
  exit /b 3
)
mkdir "%OUT%"
echo [validation] Starting %NAME%
call :gpu "%NAME% before"
if not "!ARGS:--emitter-plan-only=!"=="!ARGS!" goto execute_plan
set "MONITOR=%OUT%\gpu_samples.csv"
start "gpu-%NAME%" /b powershell -NoProfile -Command "$p=Start-Process nvidia-smi -ArgumentList '--query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits --loop=1' -RedirectStandardOutput '%MONITOR%' -WindowStyle Hidden -PassThru; $p.Id | Set-Content '%OUT%\gpu_monitor.pid'; Wait-Process -Id $p.Id"
:execute_plan
call "%PYTHON%" "%SCRIPT%" !ARGS! --output "%OUT%" > "%OUT%\run.log" 2>&1
set "RC=!ERRORLEVEL!"
set "GPU_RC=0"
set "LOG_RC=0"
if not "!ARGS:--emitter-plan-only=!"=="!ARGS!" goto skip_monitor_summary
if exist "%OUT%\gpu_monitor.pid" for /F %%P in (%OUT%\gpu_monitor.pid) do taskkill /PID %%P /T /F >nul 2>&1
call :gpu "%NAME% after"
if exist "%MONITOR%" (
  powershell -NoProfile -Command "$limit=10752; $values=@(Get-Content '%MONITOR%' | ForEach-Object { if ($_ -match ',\s*(\d+)\s*,') {[int]$Matches[1]} }); $peak=($values | Measure-Object -Maximum).Maximum; $passed=($values.Count -gt 0 -and $peak -le $limit); [ordered]@{peak_memory_mib=$peak; limit_mib=$limit; passed=$passed; sample_count=$values.Count} | ConvertTo-Json | Set-Content '%OUT%\gpu_peak.json'; if (-not $passed) { exit 6 }"
  set "GPU_RC=!ERRORLEVEL!"
) else (
  set "GPU_RC=6"
)
call :validate_log "%OUT%\run.log" "%OUT%\log_validation.json"
set "LOG_RC=!ERRORLEVEL!"
:skip_monitor_summary
if not "!RC!"=="0" (
  echo [validation] %NAME% failed with !RC!. See %OUT%\run.log
  exit /b !RC!
)
if not "!LOG_RC!"=="0" (
  echo [validation] %NAME% contains fatal log errors. See %OUT%\log_validation.json
  exit /b !LOG_RC!
)
if not "!GPU_RC!"=="0" (
  echo [validation] %NAME% exceeded the GPU memory limit or produced no samples. See %OUT%\gpu_peak.json
  exit /b !GPU_RC!
)
if not "!ARGS:--emitter-plan-only=!"=="!ARGS!" goto validation_passed
if not exist "%OUT%\run_complete.json" (
  echo [validation] %NAME% did not write run_complete.json. See %OUT%\run.log
  exit /b 4
)
:validation_passed
echo [validation] %NAME% passed
exit /b 0

:validate_log
powershell -NoProfile -Command "$lines=@(Get-Content '%~1'); $needles=@('Particle system contact buffer overflow','CUDA out of memory','CUDA OOM','non-finite','Fixed particle topology changed','Timed out waiting for capture','Capture did not produce a non-empty file','Emitter acceptance failed'); $fatal=@($lines | Select-String -SimpleMatch -Pattern $needles); $errors=@($lines | Select-String -SimpleMatch '[Error]'); [ordered]@{passed=($fatal.Count -eq 0); fatal_count=$fatal.Count; fatal_lines=@($fatal | ForEach-Object {$_.Line} | Select-Object -First 100); error_count=$errors.Count; error_lines=@($errors | ForEach-Object {$_.Line} | Select-Object -First 100)} | ConvertTo-Json -Depth 4 | Set-Content '%~2'; if ($fatal.Count -gt 0) { exit 5 }"
exit /b %ERRORLEVEL%

:gpu
>> "%ROOT%\gpu.log" echo ==== %~1 ====
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader >> "%ROOT%\gpu.log" 2>&1
exit /b 0
