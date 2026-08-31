# AI-CTO Backend (Modular Monolith v2)

Reliability-hardened FastAPI backend for the AI-CTO Platform, designed in accordance with `AICTO_Backend_Architecture_v2.md`.

## Features

- **FastAPI Modular Monolith**: High performance async API handling API & Auth, Ingestion, and FRIDAY AI LLM adapter.
- **Multi-Tenant Postgres with Row-Level Security (RLS)**: Scoped at DB layer via `app.current_business_id`.
- **Redis Multi-Purpose Engine**:
  1. Ingestion Buffer (Redis Streams)
  2. Sub-second API Response Caching (~15s TTL)
  3. FRIDAY Conversational Session Memory
  4. Distributed Job Locks (`SETNX`)
  5. LLM Endpoint Circuit Breaker & Cooldowns
  6. JWT Revocation Blacklist
- **FRIDAY LLM Adapter**: Multi-NIM priority fallback chain with half-open circuit breaker and automatic PII scrubbing.
- **Observability**: Structured JSON logs with request/business contextvars and `/api/health` diagnostic probe.

## Quickstart

### 1. Environment Setup
```bash
cd aicto-backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2. Run the Development Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 3. API Documentation
Once running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
