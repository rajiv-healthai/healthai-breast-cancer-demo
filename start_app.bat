@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_LAUNCHER="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_LAUNCHER=py -3"
if not defined PYTHON_LAUNCHER (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_LAUNCHER=python"
)

if not defined PYTHON_LAUNCHER (
    echo Python 3 was not found. Install Python 3 and try again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the app's isolated Python environment...
    %PYTHON_LAUNCHER% -m venv .venv
    if errorlevel 1 goto :failed
)

.venv\Scripts\python.exe -c "import streamlit, sklearn, numpy, pandas, joblib" >nul 2>&1
if errorlevel 1 (
    echo Installing the app requirements. This may take a few minutes the first time...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

set "CHROME_EXE="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined CHROME_EXE if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined CHROME_EXE if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%LocalAppData%\Google\Chrome\Application\chrome.exe"

echo Starting the Breast Cancer Classification Demo...
start "HealthAI Streamlit" /B ".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true

for /l %%N in (1,1,30) do (
    curl.exe --silent --fail http://127.0.0.1:8501 >nul 2>&1
    if not errorlevel 1 goto :open_app
    timeout /t 1 /nobreak >nul
)

echo The app did not become available. Check for another app using port 8501.
pause
exit /b 1

:open_app
if defined CHROME_EXE (
    start "" "%CHROME_EXE%" "http://localhost:8501"
) else (
    echo Chrome was not found in its usual install folders.
    echo Open http://localhost:8501 in your browser.
)
exit /b 0

:failed
echo.
echo The app could not start. Check the messages above and try again.
pause
exit /b 1
