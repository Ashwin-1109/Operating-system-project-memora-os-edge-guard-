@echo off
title llama.cpp Server - SmolLM2-360M (Port 8080)
color 0A
cd /d "%~dp0"

echo ======================================================================
echo    LLAMA.CPP LOCAL EDGE AI SERVER (SmolLM2-360M-Instruct)
echo    Optimized Low-Latency Configuration: Batch 512 | Slots 4
echo ======================================================================
echo.

bin\llama-server.exe -m models\SmolLM2-360M-Instruct-Q4_K_M.gguf --port 8080 -c 2048 -np 4 -b 512 -ub 512 -t 8

pause
