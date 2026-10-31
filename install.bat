@echo off
rem Starter fuer Windows. Die eigentliche Arbeit macht install.py.
rem Aufruf: install.bat [--dev] [--image] [--force]
setlocal
cd /d "%~dp0"

rem Zuerst den Python-Launcher "py", dann "python". Der Microsoft-Store-Platzhalter fuer
rem "python" startet kein echtes Python und faellt bei dieser Probe durch.
set "PY="
py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>&1 && set "PY=python"

if not defined PY (
    echo Python wurde nicht gefunden. Bitte Python 3.10 oder neuer von https://www.python.org installieren.
    set "RC=1"
) else (
    %PY% install.py %*
    call set "RC=%%ERRORLEVEL%%"
)

rem Beim Start per Doppelklick das Fenster offen lassen.
echo %cmdcmdline% | find /i "%~nx0" >nul && pause
exit /b %RC%
