# AI-CTO Backend — Render Deployment Guide

This guide details deploying the AI-CTO FastAPI backend to [Render](https://render.com) using the Dockerfile-based environment.

---

## 1. Service Architecture & Deployment Type

- **Service Type**: Web Service
- **Runtime Environment**: Docker (Render builds directly using the repository's `Dockerfile`)
- **Region**: Select the region matching your PostgreSQL and Redis instances (minimizes network latency)
- **Branch**: `main` (or designated deployment branch)

---

## 2. Pre-Deploy Command

Configure the following command in Render's **Pre-Deploy Command** field under service settings:

```bash
alembic upgrade head
```

> **Why Pre-Deploy?**  
> Render runs this command in an ephemeral container using the freshly built Docker image *before* the new application containers start serving traffic. If any database migration fails, Render automatically halts the deployment, preventing downtime or corrupted data states.

---

## 3. Environment Variables

Configure the following environment variables in the Render Dashboard (**Environment** tab).  
> **Security Notice**: Never commit sensitive secrets to git or store them in source files. Render securely injects these into the container environment at runtime.

| Variable Name | Required | Purpose & Format |
|---|---|---|
| `DATABASE_URL` | **Yes** | Connection string for PostgreSQL with asyncpg driver.<br>Format: `postgresql+asyncpg://<user>:<password>@<host>:<port>/<dbname>`<br>*(Replace `postgres://` from Render managed DB with `postgresql+asyncpg://`)* |
| `DATABASE_SYNC_URL` | **Yes** | Connection string for Alembic sync migrations.<br>Format: `postgresql://<user>:<password>@<host>:<port>/<dbname>` |
| `REDIS_URL` | **Yes** | Connection string for managed Redis instance.<br>Format: `rediss://default:<password>@<host>:<port>` (TLS enabled) or `redis://...` |
| `JWT_SECRET_KEY` | **Yes** | High-entropy secret key for signing JWT tokens (min 32 characters).<br>*(Alias: `JWT_SECRET`)* |
| `FERNET_SECRET_KEY` | **Yes** | 32 url-safe base64-encoded bytes for tenant API key encryption at rest.<br>Generate via: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `KIMI_API_KEY` | **Yes** | API key for NVIDIA NIM / Kimi Moonshot AI engine.<br>*(Alias: `MOONSHOT_API_KEY`)* |
| `KIMI_BASE_URL` | Optional | Provider endpoint (defaults to `https://api.moonshot.ai/v1`) |
| `SENTRY_DSN` | Optional | Sentry DSN URL for production error telemetry and alerting |
| `ENVIRONMENT` | **Yes** | Set to `production` |
| `PORT` | **Yes** | Set to `8000` (matches exposed Docker port and Uvicorn binding) |
| `ALLOWED_ORIGINS` | **Yes** | Comma-separated list of allowed frontend domains (e.g. `https://app.yourdomain.com`) |

---

## 4. Managed Services vs Local Compose

- **Local Development**: In local development with `docker compose`, services use internal Docker DNS names (`postgres`, `redis`).
- **Render Production**: Replace local URLs with the private or internal URLs provided by Render's managed PostgreSQL (with TimescaleDB extension enabled) and managed Redis services. These are set exclusively in Render's dashboard and must **never** be committed to the repository.

---

## 5. Health Check Endpoint

Under **Health Check Path** in Render Service Settings, specify:

```text
/api/health
```

Render will poll this endpoint during zero-downtime rolling deploys to verify the new container is fully initialized and operational before redirecting user traffic.
