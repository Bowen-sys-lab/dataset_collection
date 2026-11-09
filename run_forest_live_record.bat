@echo off
rem Forest night fire: live flight recording.
rem Same environment block and calling convention as run_collect_live.bat,
rem so the forest scene is launched exactly like the factory scenes.
rem
rem Usage: run_forest_live_record.bat [run-id] [horizontal-speed] [vertical-speed] [headless]
rem   headless: run unattended (20 frames, probe control) - smoke test only, it cannot hover
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
if "%RUNID%"=="" set RUNID=forest-live-01
set H=%2
if "%H%"=="" set H=4.0
set V=%3
if "%V%"=="" set V=2.0
set MODE=%4
if /i "%MODE%"=="headless" (set RUNMODE=--headless --frames 20 --probe-control) else (set RUNMODE=)
cd /d %PROJECT%
"%ISAAC%\python.bat" -B "scripts\start_crazyflie_dual_camera_view.py" ^
  --config "configs\forest_fire_flight_scene.yaml" ^
  --task-file "configs\forest_fire_task.yaml" ^
  --record-trajectory --run-id %RUNID% --horizontal-speed %H% --vertical-speed %V% %RUNMODE%
if errorlevel 1 exit /b %errorlevel%
echo.
echo trajectory: %PROJECT%\validation\trajectory_runs\%RUNID%\trajectory.jsonl
if /i not "%MODE%"=="headless" pause
exit /b 0
