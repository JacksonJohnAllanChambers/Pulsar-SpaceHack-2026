@echo off
REM Double-click me. Serves the contact review pages and opens your browser.
cd /d "%~dp0"

where py >nul 2>&1 && (py -3 serve.py & goto :done)
where python >nul 2>&1 && (python serve.py & goto :done)
where python3 >nul 2>&1 && (python3 serve.py & goto :done)

echo.
echo   Python was not found on this machine.
echo   Install it from https://www.python.org/downloads/ (tick "Add to PATH"),
echo   then double-click this file again. No packages are needed.
echo.
pause

:done
