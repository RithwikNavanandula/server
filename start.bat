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
echo Starting AI CCTV Server on http://0.0.0.0:5000
echo CLOUD_URL=%CLOUD_URL%
"venv\Scripts\python.exe" app.py
pause
