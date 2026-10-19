@echo off
cd /d "%~dp0"
echo Starting 4x3 Calibrated Eye Tracking Module...
"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while running the Calibrated Eye Tracking module.
    pause
)
