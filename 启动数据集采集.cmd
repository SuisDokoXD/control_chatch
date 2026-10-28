@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dpn0.ps1" %*
set "capture_exit=%errorlevel%"
if errorlevel 1 pause
exit /b %capture_exit%
