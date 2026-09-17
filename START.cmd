@echo off
setlocal
cd /d "%~dp0"
title 3D Model Image Search - Portable

echo ============================================================
echo  3D Model Image Search - Portable Launcher
echo ============================================================
echo.
echo Program folder: %CD%
echo.

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: Windows PowerShell was not found.
  echo.
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_start.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo START FAILED. Exit code: %EXIT_CODE%
  echo Please check the logs folder.
) else (
  echo Program has stopped normally.
)
echo.
pause
exit /b %EXIT_CODE%
