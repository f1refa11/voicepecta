import sqlite3
from typing import Optional

from server.config import DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user TEXT PRIMARY KEY,
                passwd TEXT NOT NULL
            )
            """
        )


def get_user(username: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute("SELECT user, passwd FROM users WHERE user = ?", (username,))
        return cur.fetchone()


def create_user(username: str, passwd_hash: str) -> bool:
    try:
        with _connect() as conn:
            conn.execute("INSERT INTO users (user, passwd) VALUES (?, ?)", (username, passwd_hash))
        return True
    except sqlite3.IntegrityError:
        return False
