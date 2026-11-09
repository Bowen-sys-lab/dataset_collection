@echo off
rem 森林火场景：静态 RGB + 红外取图（走现有 irsim 旧版与现有 scripts，不改动任何既有脚本）
rem 用法: run_forest_fire_capture.bat [out-name]
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
if "%RUNID%"=="" (
  for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set RUNID=forest-fire-static-%%i
)
cd /d %PROJECT%
"%ISAAC%\python.bat" -B "scripts\capture_crazyflie_thermal_static.py" ^
  --config "configs\forest_fire_scene.yaml" ^
  --out "%PROJECT%\validation\%RUNID%"
echo.
echo output: %PROJECT%\validation\%RUNID%
