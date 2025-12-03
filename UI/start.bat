@echo off
REM Flask 앱 시작 스크립트 (Windows)
REM Scala 자동 빌드 포함

cd /d "%~dp0"

echo Starting Flask application...
echo.

python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Failed to start application!
    pause
    exit /b 1
)

