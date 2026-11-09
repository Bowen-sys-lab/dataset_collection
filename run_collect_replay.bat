@echo off
rem Offline high-resolution replay of a recorded run -> paired RGB + thermal frames.
rem Usage: run_collect_replay.bat <run-id> [rgb-w] [rgb-h] [thermal-w] [thermal-h]
setlocal
set ISAAC=D:\isaacsim
set PROJECT=D:\IsaacSim_Demo
set USDROOT=%ISAAC%\extscache\omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312
set PXR_USD_WINDOWS_DLL_PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;%USDROOT%\bin
set PYTHONPATH=%PROJECT%;%PROJECT%\third_party\isaac-thermal-camera-simulation\src
set WARP_CACHE_PATH=%PROJECT%\validation\isaac_cache\warp
set OMNI_WARP_CACHE_PATH=%WARP_CACHE_PATH%
set PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem
set RUNID=%1
if "%RUNID%"=="" set RUNID=collect-01
if "%2"=="" (set RW=512) else (set RW=%2)
if "%3"=="" (set RH=410) else (set RH=%3)
if "%4"=="" (set TW=256) else (set TW=%4)
if "%5"=="" (set TH=205) else (set TH=%5)
cd /d %PROJECT%
"%ISAAC%\python.bat" -B "scripts\replay_trajectory_highres.py" ^
  --config "configs\model02_v7m_flight_scene.yaml" ^
  --trajectory "%PROJECT%\validation\trajectory_runs\%RUNID%\trajectory.jsonl" ^
  --out "%PROJECT%\validation\%RUNID%-hires" ^
  --rgb-width %RW% --rgb-height %RH% --thermal-width %TW% --thermal-height %TH% ^
  --rt-subframes 1
echo.
echo frames: %PROJECT%\validation\%RUNID%-hires
pause

