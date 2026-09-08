# AI-CTO Backend API (`platform-backend`) ⚡

> **Reliability-Hardened FastAPI Modular Monolith Backend v2.0.0 for AI-CTO Platform**

The **AI-CTO Backend** is an asynchronous, multi-tenant backend service designed in accordance with `AICTO_Backend_Architecture_v2.md`. It powers the **FRIDAY AI-CTO Engine**, ingests real-time telemetry streams, executes automated anomaly detection, and enforces PostgreSQL Row-Level Security (RLS).

---

## 🌟 Architecture & Key Features

- ⚡ **FastAPI Modular Monolith**: High-throughput async API endpoints handling Auth, Ingestion, Dashboard Vitals, and FRIDAY AI Hotpath.
- 🤖 **FRIDAY LLM Adapter**: Moonshot AI (Kimi) priority fallback chain (`kimi-k3`, `kimi-k2.6`, `moonshot-v1-128k`) with half-open circuit breaker, native OpenAI-compatible tool use / function calling, Server-Sent Events (SSE) streaming, PII scrubbing, and live telemetry context injection.
- 🔒 **Multi-Tenant Postgres with RLS**: Tenant data isolation at the database layer using PostgreSQL Row-Level Security (`app.current_business_id`).
- ⚡ **Redis Multi-Purpose Engine**:
  1. Ingestion Buffer (Redis Streams)
  2. Sub-second API Query Caching (~60s TTL)
  3. FRIDAY Conversational Session Memory (24h sliding window)
  4. Distributed Job Locks (`SETNX`)
  5. LLM Endpoint Cooldowns & Circuit Breaker
  6. JWT Token Revocation Blacklist
- 🛡️ **At-Rest Encryption**: Sensitive API keys and credentials encrypted using Fernet symmetric encryption key.
- 📊 **Observability & Logging**: Structured JSON logs with request ID & business contextvars (`X-Request-ID`, `X-Process-Time-Ms`).

---

## 🛠️ API Routing Table

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/friday/chat` | FRIDAY AI Conversational Hotpath (Kimi LLM + Telemetry Context) |
| `POST` | `/api/friday/chat/stream` | Real-time SSE streaming conversational token generation |
| `POST` | `/api/friday/actions/execute` | Confirmed autonomous mitigation action execution & audit logging |
| `POST` | `/api/ingestion/telemetry` | High-volume telemetry event ingestion buffer |
| `GET` | `/api/dashboard/vitals` | Real-time system vitals, latency, and active anomalies |
| `POST` | `/api/auth/login` | User authentication & JWT access token issuance |
| `GET` | `/api/health` | Comprehensive system health & diagnostic probe |

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- **Python**: `3.10` or higher
- **PostgreSQL**: `v14+` (Optional in dev fallback mode)
- **Redis**: `v6+` (Optional in dev fallback mode)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Sachin213-stack/platform-backend.git
cd platform-backend

# Create & activate Python virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Set your Kimi (Moonshot AI) API key and Fernet encryption key in `.env`:
```env
# Kimi (Moonshot AI) LLM Configuration (Sole LLM Provider)
KIMI_API_KEY="sk-your-kimi-moonshot-api-key-here"
KIMI_BASE_URL="https://api.moonshot.ai/v1"
KIMI_MODEL_PRIMARY="kimi-k3"
KIMI_MODEL_SECONDARY="kimi-k2.6"
KIMI_MODEL_FALLBACK="moonshot-v1-128k"

# Security
JWT_SECRET_KEY=aicto-super-secret-key-change-in-production-min32chars
FERNET_SECRET_KEY=ZI4lUaSB3LjrE4LrN25nmQvx98H5pc4Q29pprotD6EA=
```

> **Tip**: Generate a fresh Fernet secret key using Python:
> `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`

### 4. Run Development Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 📑 Interactive API Documentation

Once the server is running, explore and test the endpoints via:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 📂 Project Structure

```
platform-backend/
├── app/
│   ├── api/
│   │   ├── dependencies/    # Auth & RLS context injection
│   │   ├── routes/          # API Routers (Friday, Ingestion, Dashboard, Auth, Health)
│   │   └── schemas/         # Pydantic Request/Response models
│   ├── core/                # App Settings, Logging, Security (JWT/Fernet)
│   ├── db/                  # SQLAlchemy models, async sessions & migrations
│   ├── services/            # LLM Adapter, Redis Service, Ingestion Service
│   └── workers/             # Background stream telemetry workers
├── alembic/                 # Database schema migrations
├── Dockerfile               # Production container image
├── docker-compose.yml       # Local Postgres & Redis services
└── requirements.txt         # Dependencies manifest
```

---

## 📄 License

MIT © [AI-CTO Engineering Team](https://github.com/Sachin213-stack)
