@echo off
rem 夜间城市 200 m：静态 RGB + 红外取图（走现有旧版 irsim 与 scripts，不改任何既有脚本）
rem 用法: run_city_capture.bat [out-name] [camera]
rem 相机默认用交付包自带的 /World/Cameras/Middle；也可传 /World/Cameras/Start 等。
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
if "%RUNID%"=="" (
  for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set RUNID=city-200m-static-%%i
)
set CAMERA=%2
if "%CAMERA%"=="" set CAMERA=/World/Cameras/Middle
cd /d %PROJECT%
"%ISAAC%\python.bat" -B "scripts\capture_crazyflie_thermal_static.py" ^
  --config "configs\city_200m_scene.yaml" ^
  --camera-path "%CAMERA%" ^
  --out "%PROJECT%\validation\%RUNID%"
echo.
echo output: %PROJECT%\validation\%RUNID%
