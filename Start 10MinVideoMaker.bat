@echo off
setlocal
cd /d "%~dp0"
set "TENMIN_PYTHON=%~dp0..\..\..\python_embeded\python.exe"

if not exist "%TENMIN_PYTHON%" (
    echo Embedded Python was not found at:
    echo %TENMIN_PYTHON%
    echo.
    echo The launcher must remain inside ComfyUI\custom_nodes\10MinVideoMaker.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
"%TENMIN_PYTHON%" "%~dp0scripts\setup_and_start.py" --setup-only
if errorlevel 1 goto :setup_failed

"%TENMIN_PYTHON%" -u "%~dp0scripts\run_gui.py" %*
set "TENMIN_EXIT=%ERRORLEVEL%"
goto :finished

:setup_failed
set "TENMIN_EXIT=%ERRORLEVEL%"

:finished
if not "%TENMIN_EXIT%"=="0" (
    echo.
    echo 10MinVideoMaker exited with code %TENMIN_EXIT%.
    pause
)
exit /b %TENMIN_EXIT%
