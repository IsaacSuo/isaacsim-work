@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "OMNI_KIT_ACCEPT_EULA=YES"
set "PYTHON=Y:\isaacsim\python.bat"
set "SCRIPT=Y:\isaacsim_work\physx_realistic_liquid.py"
set "ROOT=Y:\isaacsim_work\output\compact_emitter_validation_v9"
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

call :gpu before
if /I "%STAGE%"=="plan" goto plan
if /I "%STAGE%"=="smoke" goto smoke
if /I "%STAGE%"=="ab" goto ab
if /I "%STAGE%"=="path" goto path
if /I "%STAGE%"=="full" goto full
if /I "%STAGE%"=="regression" goto regression
echo Usage: %~nx0 ^<plan^|smoke^|ab^|path^|full^|regression^>
exit /b 2

:plan
call :run plan_default --source-mode emitter --emitter-plan-only
if errorlevel 1 exit /b %errorlevel%
call :run plan_high_speed --source-mode emitter --emitter-speed 1.8 --frames 24 --emitter-plan-only
exit /b %errorlevel%

:smoke
call :run smoke_stream_front --source-mode emitter --spacing 0.003 --emitter-nozzle-diameter 0.024 --frames 12 --capture-every 6 --emitter-preroll-frames 30 --emitter-reserve-frames 6 --width 640 --height 360 --renderer RaytracedLighting --diagnostic-material --camera-preset stream-front --isosurface-settle-updates 4
exit /b %errorlevel%

:ab
call :run ab_default --source-mode emitter --spacing 0.0014 --emitter-nozzle-diameter 0.024 --frames 12 --capture-every 6 --emitter-preroll-frames 30 --emitter-reserve-frames 6 --width 640 --height 360 --renderer RaytracedLighting --diagnostic-material --camera-preset stream-front --isosurface-settle-updates 6
exit /b %errorlevel%

:path
call :run path_preview --source-mode emitter --spacing 0.002 --emitter-nozzle-diameter 0.024 --frames 12 --capture-every 6 --emitter-preroll-frames 30 --emitter-reserve-frames 6 --width 1280 --height 720 --renderer PathTracing --path-spp 32 --camera-preset stream-front --isosurface-settle-updates 12
exit /b %errorlevel%

:full
call :run full_default --source-mode emitter --spacing 0.0014 --emitter-nozzle-diameter 0.024 --frames 60 --capture-every 30 --emitter-preroll-frames 30 --emitter-reserve-frames 6 --width 320 --height 180 --renderer RaytracedLighting --diagnostic-material --camera-preset stream-front --isosurface-settle-updates 3
exit /b %errorlevel%

:regression
call :run regression_block --source-mode block --frames 3 --capture-every 2 --width 320 --height 180 --renderer RaytracedLighting --diagnostic-material --isosurface-settle-updates 3
if errorlevel 1 exit /b %errorlevel%
call :run regression_stream --source-mode stream --frames 3 --capture-every 2 --reservoir-settle-frames 6 --width 320 --height 180 --renderer RaytracedLighting --diagnostic-material --isosurface-settle-updates 3
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
call "%PYTHON%" "%SCRIPT%" !ARGS! --output "%OUT%" > "%OUT%\run.log" 2>&1
set "RC=%ERRORLEVEL%"
call :gpu "%NAME% after"
if not "%RC%"=="0" (
  echo [validation] %NAME% failed with %RC%. See %OUT%\run.log
  exit /b %RC%
)
if not "!ARGS:--emitter-plan-only=!"=="!ARGS!" goto validation_passed
if not exist "%OUT%\run_complete.json" (
  echo [validation] %NAME% did not write run_complete.json. See %OUT%\run.log
  exit /b 4
)
:validation_passed
echo [validation] %NAME% passed
exit /b 0

:gpu
>> "%ROOT%\gpu.log" echo ==== %~1 ====
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader >> "%ROOT%\gpu.log" 2>&1
exit /b 0
