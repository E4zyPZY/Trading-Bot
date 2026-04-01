@echo off
REM Model Update Scheduler Batch File
REM This can be scheduled via Windows Task Scheduler

echo Starting Model Update Scheduler...
echo %DATE% %TIME%

REM Change to project root directory
cd /d "%~dp0.."

REM Run the PowerShell scheduler
powershell.exe -ExecutionPolicy Bypass -File "scripts\schedule_model_updates.ps1"

echo Model Update Scheduler completed.
echo %DATE% %TIME%

REM Keep window open for 10 seconds to see results
timeout /t 10