@echo off
rem Forest fixed-view collection + validated dataset index + independent verification.
rem Factory launchers are unchanged.
rem Usage: run_forest_collection.bat [new-run-id] [baseline^|readable^|native-readable] [original^|synthetic700] [scene-config]
rem   scene-config defaults to configs\forest_fire_scene.yaml
setlocal
set "PROJECT=D:\IsaacSim_Demo"
set "ISAAC=D:\isaacsim"
set "PYTHONPATH=%PROJECT%;%PROJECT%\third_party\isaac-thermal-camera-simulation\src"
set "WARP_CACHE_PATH=%PROJECT%\validation\isaac_cache\warp"
set "OMNI_WARP_CACHE_PATH=%WARP_CACHE_PATH%"
set "RUNID=%~1"
if "%RUNID%"=="" set "RUNID=forest-%RANDOM%-%RANDOM%"
set "PROFILE=%~2"
if "%PROFILE%"=="" set "PROFILE=native-readable"
set "RANGE=%~3"
if "%RANGE%"=="" set "RANGE=original"
set "CONFIG=%~4"
if "%CONFIG%"=="" set "CONFIG=configs\forest_fire_scene.yaml"
cd /d "%PROJECT%"

echo [1/3] capture
call "%ISAAC%\python.bat" -B "scene_intake\forest_collection.py" --config "%CONFIG%" --out "validation\%RUNID%" --profile "%PROFILE%" --sensor-range "%RANGE%"
if errorlevel 1 exit /b %errorlevel%

echo [2/3] dataset index
call "%ISAAC%\python.bat" -B "scene_intake\forest_dataset.py" --capture "validation\%RUNID%" --out "validation\datasets\%RUNID%"
if errorlevel 1 exit /b %errorlevel%

echo [3/3] independent verification
call "%ISAAC%\python.bat" -B "scene_intake\verify_dataset.py" --capture "validation\%RUNID%" --json "validation\datasets\%RUNID%\verification.json"
if errorlevel 1 exit /b %errorlevel%

echo Forest capture: %PROJECT%\validation\%RUNID%
echo Dataset index: %PROJECT%\validation\datasets\%RUNID%
echo Verification: %PROJECT%\validation\datasets\%RUNID%\verification.json
exit /b 0
