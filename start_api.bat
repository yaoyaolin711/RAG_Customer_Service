@echo off
cd /d "%~dp0crm_agent\crm_agent"
set API_PORT=7120
echo ========================================
echo   RAG Agent — FastAPI
echo   模拟台  http://localhost:7120/console/
echo   文档    http://localhost:7120/docs
echo ========================================
echo.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
python main.py
pause
