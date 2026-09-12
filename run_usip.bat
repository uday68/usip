@echo off
title USIP — Underwater Sonar Intelligence Platform (SIH26057)
echo =====================================================================
echo  USIP: Underwater Sonar Intelligence Platform
echo  SIH26057: AI-Powered Marine Debris and Acoustic Anomaly Detection
echo =====================================================================
echo  Starting FastAPI Backend & Tactical Dashboard on http://127.0.0.1:8000
echo  Press Ctrl+C to stop the server.
echo =====================================================================
set PYTHONPATH=.
if exist "D:\usip_env\Scripts\python.exe" (
    D:\usip_env\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
)
pause
