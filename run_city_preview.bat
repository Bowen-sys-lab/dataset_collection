@echo off
rem 城市夜间：手动飞行 + RGB/红外实时双画面。
rem 只做预览：不记录轨迹、不保存帧。双击运行，窗口关掉即结束。
rem 附加参数会原样传给 preview，例如更清晰的画面：
rem   run_city_preview.bat --width 640 --height 512
rem 想看没有候选道具的原始城市，把下面的 CONFIG 换成 configs\city_200m_flight_scene.yaml
setlocal
set ISAAC=D:\isaacsim
set PROJECT=D:\IsaacSim_Demo
set CONFIG=configs\city_200m_scenario_scene.yaml
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
