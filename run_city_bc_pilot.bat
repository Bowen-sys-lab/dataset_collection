@echo off
rem City BC collection; forwards named options or a positional RunId.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\record_city_bc_pilot.ps1" %*
set "PILOT_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %PILOT_RESULT%
