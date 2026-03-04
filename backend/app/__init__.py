from flask import Flask

from .config import Config
from .extensions import db
from .routes import api_bp


def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    # Ensure all models are imported before create_all/init-db.
    from . import models as _models  # noqa: F401
    app.register_blueprint(api_bp)

    @app.cli.command("init-db")
    def init_db_command() -> None:
        """Create all database tables from SQLAlchemy models."""
        db.create_all()
        print("Database tables created.")

    return app
