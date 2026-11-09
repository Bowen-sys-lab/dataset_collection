@echo off
rem Compatibility entry: use the checked launcher and one collection profile.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\record_factory_bc_pilot.ps1" %*
set "PILOT_RESULT=%ERRORLEVEL%"
pause
exit /b %PILOT_RESULT%
