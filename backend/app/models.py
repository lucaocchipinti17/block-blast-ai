from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import CheckConstraint, Index, UniqueConstraint

from .extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    role = db.Column(db.String(20), nullable=False, default="user")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
    last_login_at = db.Column(db.DateTime(timezone=True), nullable=True)

    licenses = db.relationship("License", back_populates="user", cascade="all, delete-orphan")
    devices = db.relationship("Device", back_populates="user", cascade="all, delete-orphan")
    sessions = db.relationship("AuthSession", back_populates="user", cascade="all, delete-orphan")
    consumed_activation_keys = db.relationship(
        "ActivationKey",
        back_populates="consumed_by_user",
        cascade="save-update",
        foreign_keys="ActivationKey.consumed_by_user_id",
    )

    __table_args__ = (
        CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),
    )


class License(db.Model):
    __tablename__ = "licenses"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_code = db.Column(db.String(50), nullable=False, default="BLOCK_BLAST_SOLVER")
    status = db.Column(db.String(20), nullable=False, default="active")
    max_devices = db.Column(db.Integer, nullable=False, default=1)
    starts_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", back_populates="licenses")

    __table_args__ = (
        CheckConstraint("status IN ('active', 'suspended', 'cancelled', 'expired')", name="ck_licenses_status"),
        CheckConstraint("max_devices > 0", name="ck_licenses_max_devices"),
        Index("ix_licenses_user_product", "user_id", "product_code"),
    )


class ActivationKey(db.Model):
    __tablename__ = "activation_keys"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    key_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    status = db.Column(db.String(20), nullable=False, default="active")
    product_code = db.Column(db.String(50), nullable=False, default="BLOCK_BLAST_SOLVER")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)
    consumed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    consumed_by_user_id = db.Column(
        db.String(36),
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    consumed_by_user = db.relationship("User", back_populates="consumed_activation_keys")

    __table_args__ = (
        CheckConstraint("status IN ('active', 'consumed', 'revoked', 'expired')", name="ck_activation_keys_status"),
        Index("ix_activation_keys_status", "status"),
    )


class Device(db.Model):
    __tablename__ = "devices"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    hwid_hash = db.Column(db.String(128), nullable=False)
    device_label = db.Column(db.String(120), nullable=True)
    is_revoked = db.Column(db.Boolean, nullable=False, default=False)
    first_seen_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", back_populates="devices")
    sessions = db.relationship("AuthSession", back_populates="device", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("user_id", "hwid_hash", name="uq_devices_user_hwid"),
        Index("ix_devices_user_revoked", "user_id", "is_revoked"),
    )


class AuthSession(db.Model):
    __tablename__ = "auth_sessions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    device_id = db.Column(db.String(36), db.ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True)
    token_jti = db.Column(db.String(64), unique=True, nullable=False, index=True)
    status = db.Column(db.String(20), nullable=False, default="active")
    issued_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    lease_expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    last_heartbeat_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    revoked_at = db.Column(db.DateTime(timezone=True), nullable=True)
    revoke_reason = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", back_populates="sessions")
    device = db.relationship("Device", back_populates="sessions")

    __table_args__ = (
        CheckConstraint("status IN ('active', 'expired', 'revoked')", name="ck_auth_sessions_status"),
        Index("ix_auth_sessions_user_status", "user_id", "status"),
        Index("ix_auth_sessions_device_status", "device_id", "status"),
    )


class LoginAudit(db.Model):
    __tablename__ = "login_audit"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    device_id = db.Column(db.String(36), db.ForeignKey("devices.id", ondelete="SET NULL"), nullable=True, index=True)
    email = db.Column(db.String(255), nullable=True)
    event = db.Column(db.String(50), nullable=False)
    success = db.Column(db.Boolean, nullable=False, default=False)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    reason = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_login_audit_user_time", "user_id", "created_at"),
        Index("ix_login_audit_email_time", "email", "created_at"),
    )
