@echo off
rem Checked factory collection; forwards named options or a positional RunId.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\record_factory_bc_pilot.ps1" %*
set "PILOT_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %PILOT_RESULT%
