@echo off
setlocal

REM Launch the Quant News Trader main entrypoint from Task Scheduler.
cd /d "%~dp0..\quant_news_trader"

set "PYTHON_EXE="
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\venv\Scripts\python.exe"

if not defined PYTHON_EXE (
    echo ERROR: Could not find python.exe in .venv\Scripts or venv\Scripts
    exit /b 1
)

if not exist "logs" mkdir "logs"

for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "STAMP=%%i"
set "LOGFILE=logs\main_scheduler_%STAMP%.log"

echo Starting Quant News Trader main at %DATE% %TIME% > "%LOGFILE%"
echo Using interpreter: %PYTHON_EXE% >> "%LOGFILE%"

"%PYTHON_EXE%" -m src.main >> "%LOGFILE%" 2>&1
exit /b %ERRORLEVEL%