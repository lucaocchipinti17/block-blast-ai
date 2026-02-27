from flask import Blueprint, jsonify

from .extensions import db

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health() -> tuple:
    """Lightweight health check endpoint."""
    db.session.execute(db.text("SELECT 1"))
    return jsonify({"status": "ok"}), 200


@api_bp.get("/register")
def register() -> bool:
    """ Register new user """
