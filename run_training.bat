@echo off
setlocal enabledelayedexpansion

:: Navigate to project directory
cd /d "D:\Model 7.1"

:: Activate virtual environment only once
call ".venv\Scripts\activate.bat"

:RESTART
echo.
echo [INFO] === Starting main.py at %date% %time% === >> error.log

:: Run main.py and log output/errors
python main.py >> output.log 2>> error.log

:: Handle crash/restart
echo.
echo [ERROR] main.py crashed or exited unexpectedly at %date% %time% >> error.log
echo Restarting in 5 seconds...
timeout /t 5 /nobreak > nul
goto RESTART
