# Block Blast Backend (Flask + PostgreSQL)

This folder contains a starter backend for authentication, license enforcement, and single-device session control.

## Stack
- Flask
- Flask-SQLAlchemy
- PostgreSQL (via `psycopg`)

## Quick Start
1. Create and activate a virtualenv.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy env file and set values:
   ```bash
   cp .env.example .env
   ```
4. Create database schema:
   - Option A (raw SQL):
     ```bash
     psql -h localhost -U postgres -d blockblast_auth -f schema.sql
     ```
   - Option B (Flask models):
     ```bash
     flask --app app:create_app init-db
     ```
5. Run API:
   ```bash
   flask --app app:create_app run --debug
   ```

Health check:
```bash
curl http://127.0.0.1:5000/health
```

## Core Data Model
- `users`: account identity + credential hash.
- `licenses`: per-user product access and allowed device count.
- `devices`: per-user HWID hash bindings.
- `auth_sessions`: lease token records for active/login history.
- `active_session_locks`: one row per user pointing to currently active session.
- `login_audit`: immutable security/event trail for auth attempts.

## Suggested Next API Endpoints
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/heartbeat`
- `POST /auth/logout`
- `POST /devices/revoke`


ENDPOINTS:
POST /auth/register -> rehister an account to a HWID
POST /auth/login -> login, HWID hash is provided, verify user is not logged into any other device OR HWID matches
POST /auth/heartbeat -> send login + HWID, verify user is still conneted to the internet
POST /auth/logout -> post that user is not logged in anymore
POST /devices/revoke