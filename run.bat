@echo off
cd /d "%~dp0"
echo Starting Blink Detector...
"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred.
    pause
)
