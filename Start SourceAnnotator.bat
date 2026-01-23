@echo off
title SourceAnnotator Server
echo Starting SourceAnnotator...
echo.

cd /d "C:\Users\jabis\Desktop\SourceAnnotator"

:: Start the server in background and open browser
start "" cmd /c "npm run dev"

:: Wait for server to start
timeout /t 5 /nobreak > nul

:: Open in default browser
start http://localhost:5001

echo.
echo SourceAnnotator is running at http://localhost:5001
echo Press any key to stop the server...
pause > nul
