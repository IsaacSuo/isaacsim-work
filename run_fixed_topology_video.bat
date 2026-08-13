@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "STAGE=%~1"
set "JOB=%~2"
set "WRAPPER=%~dp0run_fixed_topology_video_stage.ps1"

if not exist "%WRAPPER%" exit /b 2
if not defined STAGE goto usage
if not defined JOB goto usage

if /I "%STAGE%"=="simulate-cache" (
  if "%~3"=="" goto usage
  powershell -NoProfile -File "%WRAPPER%" -Stage simulate-cache -Job "%JOB%" -RunId "%~3"
  exit /b %ERRORLEVEL%
)
if /I "%STAGE%"=="render-segment" (
  if "%~3"=="" goto usage
  if "%~4"=="" goto usage
  if "%~5"=="" goto usage
  powershell -NoProfile -File "%WRAPPER%" -Stage render-segment -Job "%JOB%" -Start %~3 -End %~4 -RunId "%~5"
  exit /b %ERRORLEVEL%
)

:usage
echo Usage:
echo   %~nx0 simulate-cache ^<job-dir^> ^<run-id^>
echo   %~nx0 render-segment ^<job-dir^> ^<start^> ^<end^> ^<run-id^>
exit /b 2
