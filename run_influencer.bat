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
  --unet-in-fp8-e4m3fn ^
  --vae-in-fp16 ^
  --attention-pytorch

pause
