@echo off
cd /d "%~dp0"
set STREAMLIT_PORT=7121
echo 启动目录: %CD%
echo Streamlit: http://localhost:%STREAMLIT_PORT%
echo.
streamlit run app.py --server.port %STREAMLIT_PORT% --server.headless true
pause
