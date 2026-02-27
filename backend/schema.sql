-- Block Blast Solver auth schema (PostgreSQL)
-- Mirrors backend/app/models.py and is suitable for initial provisioning.

BEGIN;

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ NULL,
    CONSTRAINT ck_users_role CHECK (role IN ('user', 'admin'))
);

CREATE TABLE IF NOT EXISTS licenses (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_code VARCHAR(50) NOT NULL DEFAULT 'BLOCK_BLAST_SOLVER',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    max_devices INTEGER NOT NULL DEFAULT 1,
    starts_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_licenses_status CHECK (status IN ('active', 'suspended', 'cancelled', 'expired')),
    CONSTRAINT ck_licenses_max_devices CHECK (max_devices > 0)
);

CREATE TABLE IF NOT EXISTS devices (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    hwid_hash VARCHAR(128) NOT NULL,
    device_label VARCHAR(120) NULL,
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_devices_user_hwid UNIQUE (user_id, hwid_hash)
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id VARCHAR(36) NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    token_jti VARCHAR(64) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_expires_at TIMESTAMPTZ NOT NULL,
    last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ NULL,
    revoke_reason VARCHAR(120) NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_auth_sessions_status CHECK (status IN ('active', 'expired', 'revoked'))
);

-- Exactly one active lease row per user (enforced by user_id primary key).
CREATE TABLE IF NOT EXISTS active_session_locks (
    user_id VARCHAR(36) PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(36) NOT NULL UNIQUE REFERENCES auth_sessions(id) ON DELETE CASCADE,
    locked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS login_audit (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NULL REFERENCES users(id) ON DELETE SET NULL,
    device_id VARCHAR(36) NULL REFERENCES devices(id) ON DELETE SET NULL,
    email VARCHAR(255) NULL,
    event VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    ip_address VARCHAR(64) NULL,
    user_agent VARCHAR(255) NULL,
    reason VARCHAR(255) NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);
CREATE INDEX IF NOT EXISTS ix_licenses_user_product ON licenses(user_id, product_code);
CREATE INDEX IF NOT EXISTS ix_devices_user_revoked ON devices(user_id, is_revoked);
CREATE INDEX IF NOT EXISTS ix_auth_sessions_user_status ON auth_sessions(user_id, status);
CREATE INDEX IF NOT EXISTS ix_auth_sessions_device_status ON auth_sessions(device_id, status);
CREATE INDEX IF NOT EXISTS ix_login_audit_user_time ON login_audit(user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_login_audit_email_time ON login_audit(email, created_at);

COMMIT;
