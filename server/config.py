import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DB_PATH = os.getenv("VOICEPECTA_DB_PATH", str(BASE_DIR / "voicepecta.db"))
TEMP_DIR = Path(os.getenv("VOICEPECTA_TEMP_DIR", str(BASE_DIR / "temp")))
AVATAR_DIR = Path(os.getenv("VOICEPECTA_AVATAR_DIR", str(BASE_DIR / "avatars")))

JWT_SECRET = os.getenv("VOICEPECTA_JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = int(os.getenv("VOICEPECTA_TOKEN_TTL_SECONDS", str(60 * 60 * 24 * 7)))

VOSK_MODEL_DIR = Path(os.getenv("VOICEPECTA_VOSK_MODEL_DIR", str(PROJECT_ROOT / "models" / "vosk")))
