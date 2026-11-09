@echo off
rem Live dual-camera flight + trajectory recording.
rem Usage: run_collect_live.bat [run-id]
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
cd /d %PROJECT%
"%ISAAC%\python.bat" -B "scripts\start_crazyflie_dual_camera_view.py" ^
  --config "configs\model02_v7m_flight_scene.yaml" ^
  --stage "%PROJECT%\factory\model02_thermal_v7m\model02_v7m_flight.usda" ^
  --record-trajectory --run-id %RUNID% --horizontal-speed 7.5 --vertical-speed 3.0
echo.
echo trajectory: %PROJECT%\validation\trajectory_runs\%RUNID%\trajectory.jsonl
pause



