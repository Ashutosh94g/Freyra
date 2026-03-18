@echo off
title Freyra - Influencer Edition
echo.
echo  ============================================
echo   Freyra ^| Influencer Edition
echo  ============================================
echo.

REM Change to script directory
cd /d "%~dp0"

python launch.py ^
  --preset influencer ^
  --always-high-vram ^
  --unet-in-fp8-e4m3fn

pause
