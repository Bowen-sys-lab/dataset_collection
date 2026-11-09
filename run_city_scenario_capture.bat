@echo off
rem 出题版静态取图（3 辆白车 + 3 个人 + 空调/排风热源 + 碰撞体）
rem 用法: run_city_scenario_capture.bat [out-name] [config]
rem 默认 configs\city_200m_scenario_scene.yaml；也可传 decoys / day / dark / epNNN 的配置
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
if "%RUNID%"=="" set RUNID=city-scenario-static-%RANDOM%
set CONFIG=%2
if "%CONFIG%"=="" set CONFIG=configs\city_200m_scenario_scene.yaml
cd /d %PROJECT%
"%ISAAC%\python.bat" -B "scripts\capture_crazyflie_thermal_static.py" ^
  --config "%CONFIG%" ^
  --out "%PROJECT%\validation\%RUNID%"
echo.
echo output: %PROJECT%\validation\%RUNID%
