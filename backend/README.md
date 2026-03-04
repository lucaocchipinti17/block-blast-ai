# Block Blast Backend (Simple HWID Auth)

Minimal Flask + PostgreSQL backend for:
- one-time activation keys
- account registration
- HWID-bound login
- token validation/logout

This version is intentionally simple and easy to manual-test.

## Stack
- Flask
- Flask-SQLAlchemy
- PostgreSQL (`psycopg`)

## Setup
1. Create a Python virtualenv and activate it.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure env:
   ```bash
   cp .env.example .env
   ```
4. Create tables:
   ```bash
   flask --app app:create_app init-db
   ```
5. Run API:
   ```bash
   flask --app app:create_app run --debug
   ```

## Endpoint Contract
- `GET /health`
- `POST /admin/add_key` (private; header `X-Admin-Secret`)
- `POST /auth/register` (`email`, `password`, `hwid`, `activation_key`)
- `POST /auth/login` (`email`, `password`, `hwid`)
- `POST /auth/validate` (`Authorization: Bearer <token>`, body `{ "hwid": "..." }`)
- `POST /auth/logout` (`Authorization: Bearer <token>`)
- `GET /openapi.yaml` (raw OpenAPI spec)
- `GET /docs` (Swagger UI)

## OpenAPI / Swagger
- Raw spec: `http://127.0.0.1:5000/openapi.yaml`
- Interactive docs: `http://127.0.0.1:5000/docs`

## Manual Test (curl)

Use the same `HWID` string during register/login/validate.

```bash
# 0) Health
curl http://127.0.0.1:5000/health

# 1) Create activation key (admin/private)
curl -X POST http://127.0.0.1:5000/admin/add_key \
  -H "Content-Type: application/json" \
  -H "X-Admin-Secret: change-this-admin-secret" \
  -d '{"count":1}'

# Copy activation_key from response into ACTIVATION_KEY below.

# 2) Register
curl -X POST http://127.0.0.1:5000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email":"test@example.com",
    "password":"StrongPass123!",
    "hwid":"MAC-ABC-123",
    "activation_key":"ACTIVATION_KEY_HERE"
  }'

# 3) Login
curl -X POST http://127.0.0.1:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email":"test@example.com",
    "password":"StrongPass123!",
    "hwid":"MAC-ABC-123"
  }'

# Copy access_token from login response into TOKEN below.

# 4) Validate token + device
curl -X POST http://127.0.0.1:5000/auth/validate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN_HERE" \
  -d '{"hwid":"MAC-ABC-123"}'

# 5) Logout
curl -X POST http://127.0.0.1:5000/auth/logout \
  -H "Authorization: Bearer TOKEN_HERE"
```

## Notes
- Passwords are hashed.
- Activation keys are stored as hashes (raw keys are only returned once by `/admin/add_key`).
- HWIDs are stored as peppered hashes.
- Session tokens are stored as peppered hashes.
