@echo off
setlocal
set OMNI_KIT_ACCEPT_EULA=YES
cd /d Y:\isaacsim
call Y:\isaacsim\python.bat Y:\isaacsim_work\physx_water_clean.py %*
endlocal
