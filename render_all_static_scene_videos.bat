@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "ISAAC_PYTHON=Y:\isaacsim\python.bat"
set "RUNNER=%~dp0tools\run_static_scene_videos.py"
if not exist "%ISAAC_PYTHON%" exit /b 2
if not exist "%RUNNER%" exit /b 2
call "%ISAAC_PYTHON%" "%RUNNER%" %*
exit /b %ERRORLEVEL%
