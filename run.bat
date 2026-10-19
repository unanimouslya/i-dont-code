@echo off
cd /d "%~dp0"
echo Starting Posture and Movement Module...
"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while running the Posture and Movement module.
    pause
)
