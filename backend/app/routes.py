from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request, Response
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db
from .models import ActivationKey, AuthSession, Device, License, LoginAudit, User, utcnow

api_bp = Blueprint("api", __name__)
OPENAPI_SPEC_PATH = Path(__file__).resolve().parent.parent / "openapi.yaml"


def _json_error(message: str, status: int = 400) -> tuple:
    return jsonify({"ok": False, "error": message}), status


def _hash_with_pepper(value: str, pepper: str) -> str:
    payload = f"{pepper}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_hwid(hwid_raw: str) -> str:
    return _hash_with_pepper(hwid_raw.strip(), current_app.config["HWID_PEPPER"])


def _hash_activation_key(key_raw: str) -> str:
    return _hash_with_pepper(key_raw.strip(), current_app.config["ACTIVATION_KEY_PEPPER"])


def _hash_session_token(token_raw: str) -> str:
    return _hash_with_pepper(token_raw.strip(), current_app.config["SESSION_TOKEN_PEPPER"])


def _get_json() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return {}
    return data


def _required_fields(data: dict[str, Any], *names: str) -> tuple[bool, list[str]]:
    missing = [n for n in names if not str(data.get(n, "")).strip()]
    return len(missing) == 0, missing


def _admin_authorized() -> bool:
    provided = request.headers.get("X-Admin-Secret", "")
    expected = str(current_app.config["ADMIN_API_SECRET"])
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def _bearer_token() -> str:
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header.lower().startswith("bearer "):
        return ""
    return auth_header[7:].strip()


def _license_is_active(lic: License | None) -> bool:
    if lic is None or lic.status != "active":
        return False
    if lic.expires_at is not None and lic.expires_at <= utcnow():
        return False
    return True


def _active_license_for_user(user_id: str) -> License | None:
    licenses = (
        License.query.filter_by(user_id=user_id)
        .order_by(License.created_at.desc())
        .all()
    )
    for lic in licenses:
        if _license_is_active(lic):
            return lic
    return None


def _audit(
    *,
    event: str,
    success: bool,
    email: str | None = None,
    user_id: str | None = None,
    device_id: str | None = None,
    reason: str | None = None,
) -> None:
    db.session.add(
        LoginAudit(
            event=event,
            success=success,
            email=email,
            user_id=user_id,
            device_id=device_id,
            reason=reason,
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent", ""),
        )
    )


@api_bp.get("/health")
def health() -> tuple:
    db.session.execute(db.text("SELECT 1"))
    return jsonify({"ok": True, "status": "ok"}), 200


@api_bp.get("/openapi.yaml")
def openapi_yaml() -> tuple | Response:
    if not OPENAPI_SPEC_PATH.exists():
        return _json_error("OpenAPI spec not found.", 500)
    return Response(OPENAPI_SPEC_PATH.read_text(encoding="utf-8"), mimetype="application/yaml")


@api_bp.get("/docs")
def docs() -> Response:
    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Block Blast API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
    <style>
      body { margin: 0; background: #fafafa; }
      #swagger-ui { max-width: 1200px; margin: 0 auto; }
    </style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: '/openapi.yaml',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis],
      });
    </script>
  </body>
</html>
""".strip()
    return Response(html, mimetype="text/html")


@api_bp.post("/admin/add_key")
def add_key() -> tuple:
    """
    Private endpoint used by your website/backend after payment.
    Creates one or more activation keys and stores only key hashes.
    """
    if not _admin_authorized():
        return _json_error("Unauthorized", 401)

    data = _get_json()
    try:
        count = int(data.get("count", 1))
    except Exception:  # noqa: BLE001
        return _json_error("count must be an integer in [1, 20].", 400)

    if count < 1 or count > 20:
        return _json_error("count must be in [1, 20].", 400)

    keys_out: list[str] = []
    for _ in range(count):
        raw_key = f"BB-{secrets.token_hex(12).upper()}"
        key_hash = _hash_activation_key(raw_key)
        db.session.add(ActivationKey(key_hash=key_hash, status="active"))
        keys_out.append(raw_key)

    db.session.commit()
    if count == 1:
        return jsonify({"ok": True, "activation_key": keys_out[0]}), 201
    return jsonify({"ok": True, "activation_keys": keys_out}), 201


@api_bp.post("/auth/register")
def register() -> tuple:
    """
    Register account + bind first device using one-time activation key.
    Request JSON: {email, password, hwid, activation_key}
    """
    data = _get_json()
    ok, missing = _required_fields(data, "email", "password", "hwid", "activation_key")
    if not ok:
        return _json_error(f"Missing required fields: {', '.join(missing)}", 400)

    email = str(data["email"]).strip().lower()
    password = str(data["password"])
    hwid_raw = str(data["hwid"]).strip()
    activation_key_raw = str(data["activation_key"]).strip()

    if len(password) < 8:
        return _json_error("Password must be at least 8 characters.", 400)

    if User.query.filter_by(email=email).first() is not None:
        return _json_error("Email is already registered.", 409)

    key_hash = _hash_activation_key(activation_key_raw)
    key_row = ActivationKey.query.filter_by(key_hash=key_hash).first()
    if key_row is None or key_row.status != "active":
        return _json_error("Invalid activation key.", 400)

    now = utcnow()
    if key_row.expires_at is not None and key_row.expires_at <= now:
        key_row.status = "expired"
        db.session.commit()
        return _json_error("Activation key expired.", 400)

    hwid_hash = _hash_hwid(hwid_raw)

    try:
        user = User(email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.flush()

        db.session.add(
            License(
                user_id=user.id,
                status="active",
                max_devices=1,
            )
        )

        db.session.add(
            Device(
                user_id=user.id,
                hwid_hash=hwid_hash,
                device_label=str(data.get("device_label", "")).strip() or None,
            )
        )

        key_row.status = "consumed"
        key_row.consumed_at = now
        key_row.consumed_by_user_id = user.id

        _audit(event="register", success=True, email=email, user_id=user.id)
        db.session.commit()
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        return _json_error(f"Registration failed: {exc}", 500)

    return jsonify({"ok": True, "message": "Registration successful."}), 201


@api_bp.post("/auth/login")
def login() -> tuple:
    """
    Login with email/password/HWID.
    Request JSON: {email, password, hwid}
    """
    data = _get_json()
    ok, missing = _required_fields(data, "email", "password", "hwid")
    if not ok:
        return _json_error(f"Missing required fields: {', '.join(missing)}", 400)

    email = str(data["email"]).strip().lower()
    password = str(data["password"])
    hwid_hash = _hash_hwid(str(data["hwid"]))

    user = User.query.filter_by(email=email).first()
    if user is None or not user.is_active or not check_password_hash(user.password_hash, password):
        _audit(event="login", success=False, email=email, reason="invalid_credentials")
        db.session.commit()
        return _json_error("Invalid credentials or device.", 401)

    license_row = _active_license_for_user(user.id)
    if not _license_is_active(license_row):
        _audit(event="login", success=False, email=email, user_id=user.id, reason="license_inactive")
        db.session.commit()
        return _json_error("License inactive.", 403)

    device = Device.query.filter_by(user_id=user.id, hwid_hash=hwid_hash, is_revoked=False).first()
    if device is None:
        _audit(event="login", success=False, email=email, user_id=user.id, reason="hwid_not_allowed")
        db.session.commit()
        return _json_error("Invalid credentials or device.", 401)

    now = utcnow()
    ttl_hours = int(current_app.config["SESSION_TTL_HOURS"])
    session_expires = now + timedelta(hours=max(1, ttl_hours))
    raw_token = f"bbs_{secrets.token_urlsafe(32)}"
    token_hash = _hash_session_token(raw_token)

    db.session.add(
        AuthSession(
            user_id=user.id,
            device_id=device.id,
            token_jti=token_hash,
            status="active",
            lease_expires_at=session_expires,
            last_heartbeat_at=now,
        )
    )
    user.last_login_at = now
    device.last_seen_at = now
    _audit(event="login", success=True, email=email, user_id=user.id, device_id=device.id)
    db.session.commit()

    return (
        jsonify(
            {
                "ok": True,
                "access_token": raw_token,
                "token_type": "bearer",
                "expires_at": session_expires.isoformat(),
            }
        ),
        200,
    )


@api_bp.post("/auth/validate")
def validate() -> tuple:
    """
    Validate active token + device binding.
    Header: Authorization: Bearer <token>
    Request JSON: {hwid}
    """
    token = _bearer_token()
    if not token:
        return _json_error("Missing bearer token.", 401)

    data = _get_json()
    ok, missing = _required_fields(data, "hwid")
    if not ok:
        return _json_error(f"Missing required fields: {', '.join(missing)}", 400)

    hwid_hash = _hash_hwid(str(data["hwid"]))
    token_hash = _hash_session_token(token)
    now = utcnow()

    sess = AuthSession.query.filter_by(token_jti=token_hash, status="active").first()
    if sess is None:
        return _json_error("Invalid or expired token.", 401)

    if sess.lease_expires_at <= now:
        sess.status = "expired"
        db.session.commit()
        return _json_error("Invalid or expired token.", 401)

    user = User.query.get(sess.user_id)
    if user is None or not user.is_active:
        return _json_error("Account inactive.", 403)

    license_row = _active_license_for_user(user.id)
    if not _license_is_active(license_row):
        return _json_error("License inactive.", 403)

    device = Device.query.filter_by(id=sess.device_id, user_id=user.id, is_revoked=False).first()
    if device is None or device.hwid_hash != hwid_hash:
        return _json_error("Device not allowed.", 401)

    device.last_seen_at = now
    db.session.commit()
    return (
        jsonify(
            {
                "ok": True,
                "valid": True,
                "user_id": user.id,
                "email": user.email,
                "expires_at": sess.lease_expires_at.isoformat(),
            }
        ),
        200,
    )


@api_bp.post("/auth/logout")
def logout() -> tuple:
    """
    Revoke current token.
    Header: Authorization: Bearer <token>
    """
    token = _bearer_token()
    if not token:
        return _json_error("Missing bearer token.", 401)

    token_hash = _hash_session_token(token)
    sess = AuthSession.query.filter_by(token_jti=token_hash, status="active").first()
    if sess is None:
        return jsonify({"ok": True, "message": "Already logged out."}), 200

    sess.status = "revoked"
    sess.revoked_at = utcnow()
    sess.revoke_reason = "manual_logout"
    db.session.commit()
    return jsonify({"ok": True, "message": "Logged out."}), 200
