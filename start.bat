@echo off
cd /d "%~dp0"
if exist .env (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%~A"=="" set "%%~A=%%~B"
  )
)
if not exist "venv\Scripts\python.exe" (
  echo Run setup.ps1 first.
  pause
  exit /b 1
)
if not defined PORT set "PORT=5000"
echo Starting AI CCTV Server on http://0.0.0.0:%PORT%
echo CLOUD_URL=%CLOUD_URL%
echo PORT=%PORT%
"venv\Scripts\python.exe" app.py
pause
