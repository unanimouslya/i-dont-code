# 4x3 Calibrated Eye Tracking Module - PowerShell Launcher
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "Starting 4x3 Calibrated Eye Tracking Module..." -ForegroundColor Cyan
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" main.py
