import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DB_BACKEND = os.getenv("VOICEPECTA_DB_BACKEND", "sqlite").strip().lower()
if DB_BACKEND == "mariadb":
    DB_BACKEND = "mysql"

DB_PATH = os.getenv("VOICEPECTA_DB_PATH", str(BASE_DIR / "voicepecta.db"))
MYSQL_HOST = os.getenv("VOICEPECTA_MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("VOICEPECTA_MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("VOICEPECTA_MYSQL_USER", "voicepecta")
MYSQL_PASSWORD = os.getenv("VOICEPECTA_MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("VOICEPECTA_MYSQL_DATABASE", "voicepecta")

TEMP_DIR = Path(os.getenv("VOICEPECTA_TEMP_DIR", str(BASE_DIR / "temp")))
AVATAR_DIR = Path(os.getenv("VOICEPECTA_AVATAR_DIR", str(BASE_DIR / "avatars")))

JWT_SECRET = os.getenv("VOICEPECTA_JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = int(os.getenv("VOICEPECTA_TOKEN_TTL_SECONDS", str(60 * 60 * 24 * 7)))

TRIAL_DURATION_HOURS = int(os.getenv("VOICEPECTA_TRIAL_DURATION_HOURS", "720"))
WEEKLY_TOKEN_LIMIT = int(os.getenv("VOICEPECTA_WEEKLY_TOKEN_LIMIT", "2000"))
TOKEN_RESET_INTERVAL_SECONDS = int(os.getenv("VOICEPECTA_TOKEN_RESET_INTERVAL_SECONDS", str(7 * 24 * 60 * 60)))
WHISPER_SEGMENTS_WITH_TIMESTAMPS = _env_bool("VOICEPECTA_WHISPER_SEGMENTS_WITH_TIMESTAMPS", True)

VOSK_MODEL_DIR = Path(os.getenv("VOICEPECTA_VOSK_MODEL_DIR", str(PROJECT_ROOT / "models" / "vosk")))
