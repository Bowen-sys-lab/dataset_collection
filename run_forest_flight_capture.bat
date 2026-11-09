@echo off
rem 森林火场景：飞一小段 + 出 RGB/红外配对（一条命令）
rem
rem 为什么录两次：既有回放会校验"录制时的场景哈希"，而透明火舌在 RGB 里要保留、
rem 在红外里必须换成不透明热代面，所以 RGB 与红外各用一份场景、各录一条轨迹。
rem 探针控制是确定性的，两条轨迹逐帧位置一致（实测差 0.0），因此合并是精确对齐的。
rem
rem 用法: run_forest_flight_capture.bat <run-id> [frames] [horizontal-speed]
setlocal
set "PROJECT=D:\IsaacSim_Demo"
set "ISAAC=D:\isaacsim"
set "USDROOT=%ISAAC%\extscache\omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312"
set "PXR_USD_WINDOWS_DLL_PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;%USDROOT%\bin"
set "PYTHONPATH=%PROJECT%;%PROJECT%\third_party\isaac-thermal-camera-simulation\src"
set "WARP_CACHE_PATH=%PROJECT%\validation\isaac_cache\warp"
set "OMNI_WARP_CACHE_PATH=%WARP_CACHE_PATH%"
set "PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem"
set "RUNID=%~1"
if "%RUNID%"=="" set "RUNID=forest-flight-%RANDOM%"
set "FRAMES=%~2"
if "%FRAMES%"=="" set "FRAMES=120"
set "SPEED=%~3"
if "%SPEED%"=="" set "SPEED=2.0"
cd /d "%PROJECT%"

echo [1/6] record RGB trajectory (normal flame cards)
call "%ISAAC%\python.bat" -B "scripts\start_crazyflie_dual_camera_view.py" ^
  --config "configs\forest_fire_flight_scene.yaml" --headless --frames %FRAMES% ^
  --record-trajectory --run-id "%RUNID%-rgb" --probe-control --horizontal-speed %SPEED%
if errorlevel 1 exit /b %errorlevel%

echo [2/6] record IR trajectory (baked flame silhouette)
call "%ISAAC%\python.bat" -B "scripts\start_crazyflie_dual_camera_view.py" ^
  --config "configs\forest_fire_flight_flame_scene.yaml" --headless --frames %FRAMES% ^
  --record-trajectory --run-id "%RUNID%-ir" --probe-control --horizontal-speed %SPEED%
if errorlevel 1 exit /b %errorlevel%

echo [3/6] replay RGB
call "%ISAAC%\python.bat" -X utf8 "scene_intake\run_with_local_assets.py" "scripts\replay_trajectory_highres.py" -- ^
  --config "configs\forest_fire_flight_scene.yaml" ^
  --trajectory "validation\trajectory_runs\%RUNID%-rgb\trajectory.jsonl" ^
  --out "validation\%RUNID%-rgb" --rgb-width 640 --rgb-height 512 --thermal-width 320 --thermal-height 256
if errorlevel 1 exit /b %errorlevel%

echo [4/6] replay IR (flame visible)
call "%ISAAC%\python.bat" -X utf8 "scene_intake\run_with_local_assets.py" "scripts\replay_trajectory_highres.py" -- ^
  --config "configs\forest_fire_flight_flame_scene.yaml" ^
  --trajectory "validation\trajectory_runs\%RUNID%-ir\trajectory.jsonl" ^
  --out "validation\%RUNID%-thermal" --rgb-width 640 --rgb-height 512 --thermal-width 320 --thermal-height 256
if errorlevel 1 exit /b %errorlevel%

echo [5/6] merge RGB + IR
call "%ISAAC%\python.bat" -B "scene_intake\merge_replay_passes.py" ^
  --rgb-run "validation\%RUNID%-rgb" --thermal-run "validation\%RUNID%-thermal" ^
  --out "validation\%RUNID%-merged"
if errorlevel 1 exit /b %errorlevel%

echo [6/6] task evaluation (informational: FAIL is a valid observation, not a broken run)
call "%ISAAC%\python.bat" -B "scene_intake\evaluate_forest_task.py" ^
  --trajectory "validation\trajectory_runs\%RUNID%-rgb\trajectory.jsonl" ^
  --capture "validation\%RUNID%-merged" ^
  --task "configs\forest_fire_task.yaml" ^
  --json "validation\%RUNID%-merged\task_evaluation.json"

echo.
echo merged frames: %PROJECT%\validation\%RUNID%-merged
echo task evaluation: %PROJECT%\validation\%RUNID%-merged\task_evaluation.json
exit /b 0
