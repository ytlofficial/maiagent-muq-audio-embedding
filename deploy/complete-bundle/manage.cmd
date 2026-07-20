@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0manage.ps1" %*
exit /b %ERRORLEVEL%
