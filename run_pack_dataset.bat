@echo off
rem Pack one replay run into a queryable dataset entry.
rem Usage: run_pack_dataset.bat <run-id> [replay-suffix]
setlocal
set ISAAC=D:\isaacsim
set PROJECT=D:\IsaacSim_Demo
set USDROOT=%ISAAC%\extscache\omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312
set PXR_USD_WINDOWS_DLL_PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;%USDROOT%\bin
set PYTHONPATH=%PROJECT%;%PROJECT%\third_party\isaac-thermal-camera-simulation\src
set PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem
set RUNID=%1
if "%RUNID%"=="" set RUNID=collect-003
if "%2"=="" (set REPLAY=%PROJECT%\validation\%RUNID%-full) else (set REPLAY=%PROJECT%\validation\%RUNID%-%2)
set EVAL=%PROJECT%\validation\%RUNID%-eval\factory_evaluation.json
cd /d %PROJECT%
if not exist "%REPLAY%" (echo replay dir not found: %REPLAY% & pause & exit /b 1)
"%ISAAC%\python.bat" -B "scripts\build_dataset.py" --run "%PROJECT%\validation\trajectory_runs\%RUNID%" --replay "%REPLAY%" --config "configs\model02_v7m_flight_scene.yaml" --eval "%EVAL%" --out "%PROJECT%\validation\datasets\%RUNID%"
echo.
echo dataset: %PROJECT%\validation\datasets\%RUNID%
pause