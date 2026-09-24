@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe"
set "APP_SCRIPT=%PROJECT_DIR%2026-09-24_multimodal-viewer_v1.0.py"

if not exist "%PYTHON_EXE%" (
    echo Python environment not found:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

if not exist "%APP_SCRIPT%" (
    echo Qt showcase script not found:
    echo %APP_SCRIPT%
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" "%APP_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Qt showcase exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
