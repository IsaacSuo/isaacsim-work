@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "ISAAC_PYTHON=Y:\isaacsim\python.bat"
set "SCRIPT=%~dp0cloth_flag_hero.py"
set "OUTPUT=%~dp0output\cloth_flag_video"
set "FRAME_DIR=%OUTPUT%\video_frames"
set "VIDEO=%OUTPUT%\cloth_flag_hero.mp4"
set "EXPECTED_FRAMES=240"

if not exist "%ISAAC_PYTHON%" (
  echo [video] Missing Isaac Sim Python: %ISAAC_PYTHON%
  exit /b 2
)
if not exist "%SCRIPT%" (
  echo [video] Missing hero script: %SCRIPT%
  exit /b 2
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [video] ffmpeg is not available in Windows PATH.
  exit /b 2
)
if /I "%~1"=="--check" (
  call "%ISAAC_PYTHON%" "%SCRIPT%" --help >nul
  if errorlevel 1 exit /b 1
  echo [video] Check passed: Isaac Python, hero script, and FFmpeg are available.
  exit /b 0
)

echo [video] Script: %SCRIPT%
echo [video] Output: %OUTPUT%
echo [video] Rendering %EXPECTED_FRAMES% physics frames at 960x960, PathTracing 32 spp.

call "%ISAAC_PYTHON%" "%SCRIPT%" ^
  --frames %EXPECTED_FRAMES% ^
  --substeps 2 ^
  --width 960 ^
  --height 960 ^
  --renderer PathTracing ^
  --path-spp 32 ^
  --render-settle 32 ^
  --render-video-frames ^
  --video-frames-dir video_frames ^
  --output "%OUTPUT%"

if errorlevel 1 (
  echo [video] Isaac Sim render failed. See %OUTPUT%\run.log
  exit /b 1
)

set "FRAME_COUNT=0"
for /f %%C in ('dir /b /a-d "%FRAME_DIR%\frame_*.png" 2^>nul ^| find /c /v ""') do set "FRAME_COUNT=%%C"
if not "%FRAME_COUNT%"=="%EXPECTED_FRAMES%" (
  echo [video] Expected %EXPECTED_FRAMES% frames but found %FRAME_COUNT%.
  exit /b 1
)

echo [video] Encoding %FRAME_COUNT% frames to H.264 MP4 at 60 fps.
ffmpeg -hide_banner -loglevel info -y ^
  -framerate 60 ^
  -start_number 0 ^
  -i "%FRAME_DIR%\frame_%%06d.png" ^
  -c:v libx264 ^
  -preset slow ^
  -crf 16 ^
  -pix_fmt yuv420p ^
  -movflags +faststart ^
  "%VIDEO%"

if errorlevel 1 (
  echo [video] FFmpeg encoding failed.
  exit /b 1
)
if not exist "%VIDEO%" (
  echo [video] FFmpeg returned success but the MP4 is missing.
  exit /b 1
)

for %%F in ("%VIDEO%") do echo [video] Complete: %%~fF ^(%%~zF bytes^)
exit /b 0
