@echo off
rem 出题版飞一小段 + 高分辨率回放；可带一条任务文件（默认 N3）
rem 用法: run_city_scenario_flight.bat [run-id] [task-file] [frames] [speed]
setlocal
set ISAAC=D:\isaacsim
set PROJECT=D:\IsaacSim_Demo
set USDROOT=%ISAAC%\extscache\omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312
set PXR_USD_WINDOWS_DLL_PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;%USDROOT%\bin
set PYTHONPATH=%PROJECT%;%PROJECT%\third_party\isaac-thermal-camera-simulation\src
set WARP_CACHE_PATH=%PROJECT%\validation\isaac_cache\warp
set OMNI_WARP_CACHE_PATH=%WARP_CACHE_PATH%
rem 机体 payload（cf2x.usd）已缓存到本机；换机器时先设一次 HTTP_PROXY/HTTPS_PROXY，
rem 让 Isaac 能拉取 https S3 资源，缓存后可去掉。 
set PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem
set RUNID=%1
if "%RUNID%"=="" set RUNID=city-scenario-flight-%RANDOM%
set TASKFILE=%2
if "%TASKFILE%"=="" set TASKFILE=configs\city_200m_task_n3.yaml
set FRAMES=%3
if "%FRAMES%"=="" set FRAMES=180
set SPEED=%4
if "%SPEED%"=="" set SPEED=3.0
cd /d %PROJECT%
echo [1/2] record trajectory with task %TASKFILE%
"%ISAAC%\python.bat" -B "scripts\start_crazyflie_dual_camera_view.py" ^
  --config "configs\city_200m_scenario_scene.yaml" --headless --frames %FRAMES% ^
  --record-trajectory --run-id "%RUNID%" --probe-control --horizontal-speed %SPEED% ^
  --task-file "%TASKFILE%"
if errorlevel 1 exit /b %errorlevel%
echo [2/2] replay high resolution
"%ISAAC%\python.bat" -X utf8 "scripts\replay_trajectory_highres.py" ^
  --config "configs\city_200m_scenario_scene.yaml" ^
  --trajectory "validation\trajectory_runs\%RUNID%\trajectory.jsonl" ^
  --out "validation\%RUNID%" --rgb-width 640 --rgb-height 512 --thermal-width 320 --thermal-height 256
if errorlevel 1 exit /b %errorlevel%
echo.
echo trajectory: %PROJECT%\validation\trajectory_runs\%RUNID%
echo frames:     %PROJECT%\validation\%RUNID%
