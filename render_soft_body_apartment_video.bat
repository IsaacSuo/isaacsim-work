@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "ISAAC_PYTHON=Y:\isaacsim\python.bat"
set "SCRIPT=%~dp0soft_body_bounce_hero.py"
set "OUTPUT=%~dp0output\soft_body_apartment_5s"
set "FRAME_DIR=%OUTPUT%\video_frames"
set "VIDEO=%OUTPUT%\soft_body_apartment_5s.mp4"
set "EXPECTED_FRAMES=300"
if not exist "%ISAAC_PYTHON%" exit /b 2
if not exist "%SCRIPT%" exit /b 2
where ffmpeg >nul 2>nul
if errorlevel 1 exit /b 2
call "%ISAAC_PYTHON%" "%SCRIPT%" ^
  --frames %EXPECTED_FRAMES% ^
  --substeps 4 ^
  --model "Y:\isaacsim_work\assets\soft_body_elephant.stl" ^
  --model-height 1.45 ^
  --drop-height 3.0 ^
  --environment-usd "Y:\scenes\apartment\apartment_sim.usda" ^
  --support-top-y -0.489954 ^
  --spawn-x 0.0 ^
  --spawn-z 0.0 ^
  --camera-eye 5.5 4.0 8.1 ^
  --camera-target 0.0 1.75 0.0 ^
  --width 960 ^
  --height 960 ^
  --renderer PathTracing ^
  --path-spp 32 ^
  --render-settle 32 ^
  --render-video-frames ^
  --video-frames-dir video_frames ^
  --output "%OUTPUT%"
if errorlevel 1 exit /b 1
set "FRAME_COUNT=0"
for /f %%C in ('dir /b /a-d "%FRAME_DIR%\frame_*.png" 2^>nul ^| find /c /v ""') do set "FRAME_COUNT=%%C"
if not "%FRAME_COUNT%"=="%EXPECTED_FRAMES%" exit /b 1
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
if errorlevel 1 exit /b 1
for %%F in ("%VIDEO%") do echo [video] Complete: %%~fF ^(%%~zF bytes^)
exit /b 0
