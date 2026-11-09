@echo off
rem City v2 tasks with 2026-11-08 material-partition v3 thermal scene.
rem 只做预览：不记录轨迹、不保存帧。双击运行，窗口关掉即结束。
rem 附加参数原样传给 preview，例如更清晰的画面：
rem   run_city_v2_preview.bat --width 640 --height 512
setlocal
set ISAAC=D:\isaacsim
set PROJECT=D:\IsaacSim_Demo
set CONFIG=configs\city_200m_v3_scene.yaml
set USDROOT=%ISAAC%\extscache\omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312
set PXR_USD_WINDOWS_DLL_PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;%USDROOT%\bin
set PYTHONPATH=%PROJECT%;%PROJECT%\third_party\isaac-thermal-camera-simulation\src
set WARP_CACHE_PATH=%PROJECT%\validation\isaac_cache\warp
set OMNI_WARP_CACHE_PATH=%WARP_CACHE_PATH%
set PATH=%ISAAC%\kit;%ISAAC%\kit\bin;%ISAAC%\kit\python;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem
cd /d %PROJECT%
"%ISAAC%\python.bat" -B "scripts\start_crazyflie_dual_camera_view.py" ^
  --config "%CONFIG%" ^
  --preview-mode live ^
  --live-refresh paired ^
  --no-save ^
  --horizontal-speed 3 ^
  --vertical-speed 1 ^
  %*
