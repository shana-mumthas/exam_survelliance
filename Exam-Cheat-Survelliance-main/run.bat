@echo off
echo ==========================================
echo Starting Exam Surveillance System...
echo URL: http://localhost:8000
echo ==========================================
echo.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
