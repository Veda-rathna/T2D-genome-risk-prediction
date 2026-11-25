@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   T2D Genome Risk Prediction - DBT Research Demonstration
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not available in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

echo Checking Streamlit installation...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo.
echo Launching local Streamlit server at http://localhost:8501 ...
echo Press Ctrl+C in this window to stop the server.
echo.
python -m streamlit run app.py --server.port 8501 --server.headless false

endlocal
