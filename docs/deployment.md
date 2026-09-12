# Deployment Guide & Production Execution

**Platform**: USIP — Underwater Sonar Intelligence Platform (SIH26057)

---

## 1. Local Production Deployment (Windows / Linux)

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (e.g. NVIDIA RTX 3050) or Multi-core CPU
- Node.js 18+ (optional, since the integrated dashboard is served directly by FastAPI)

### Quick Start
1. Ensure the virtual environment is activated:
   ```powershell
   D:\usip_env\Scripts\activate
   ```
2. Launch the FastAPI server:
   ```powershell
   $env:PYTHONPATH="."
   D:\usip_env\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
3. Access the dashboard:
   Open browser at **`http://localhost:8000`**.

---

## 2. Docker Containerized Deployment

A multi-stage containerized setup is provided via `docker/Dockerfile` and `docker-compose.yml`.

### Build & Run
```bash
docker-compose up --build -d
```

### Services Started:
1. **`usip-backend`**: FastAPI REST API on port `8000`.
2. **`usip-db`**: PostgreSQL 16 with PostGIS extensions on port `5432`.

