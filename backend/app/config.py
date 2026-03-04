import os


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/blockblast_auth",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ADMIN_API_SECRET = os.getenv("ADMIN_API_SECRET", "change-me")
    HWID_PEPPER = os.getenv("HWID_PEPPER", "hwid-pepper-change-me")
    ACTIVATION_KEY_PEPPER = os.getenv("ACTIVATION_KEY_PEPPER", "activation-pepper-change-me")
    SESSION_TOKEN_PEPPER = os.getenv("SESSION_TOKEN_PEPPER", "session-pepper-change-me")
    SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
