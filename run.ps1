# Typing Dynamics Module - PowerShell Launcher
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "Starting Typing Dynamics Module..." -ForegroundColor Cyan
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" main.py
