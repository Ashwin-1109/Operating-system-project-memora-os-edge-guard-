@echo off
title MemoraOS EdgeGuard - Academic Demonstration
color 0B
cd /d "%~dp0"

echo ======================================================================
echo    MEMORAOS EDGEGUARD: REAL-TIME EDGE AI MEMORY MANAGEMENT
echo    Academic Proof of Concept Demonstration
echo ======================================================================
echo.

echo [1/3] Verifying Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo [2/3] Checking dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Could not install some packages, continuing anyway...
)

echo.
echo [2.5/3] Checking local Edge AI model server - llama.cpp...
tasklist /FI "IMAGENAME eq llama-server.exe" 2>NUL | find /I /N "llama-server.exe">NUL
if not errorlevel 1 (
    echo [OK] llama-server.exe is already active and listening on port 8080.
) else (
    echo [STARTING] Launching local llama-server on port 8080 with 8 compute threads...
    start "llama-server" /min cmd /c run_llama_server.bat
    timeout /t 3 /nobreak >nul
)

echo.
echo [3/3] Launching MemoraOS EdgeGuard Dashboard...
echo The dashboard will automatically open in your default browser at http://localhost:8501
echo.

:: Launch Streamlit server and automatically open browser when ready
python -m streamlit run app.py --server.port 8501 --server.headless false --browser.serverAddress localhost

pause
