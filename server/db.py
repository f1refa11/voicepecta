import sqlite3
import time
from typing import Optional, Tuple

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
        _ensure_columns(conn)
        conn.commit()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in cur.fetchall()}
    if "email" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
    # enforce uniqueness only when email is set, via partial index
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email) WHERE email IS NOT NULL"
    )
    if "profile_pic" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
    if "daily_used" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN daily_used INTEGER DEFAULT 0")
    if "daily_reset_at" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN daily_reset_at INTEGER DEFAULT 0")


def get_user(username: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT user, passwd, email, profile_pic, daily_used, daily_reset_at FROM users WHERE user = ?",
            (username,),
        )
        return cur.fetchone()


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT user, passwd, email, profile_pic, daily_used, daily_reset_at FROM users WHERE email = ?",
            (email,),
        )
        return cur.fetchone()


def create_user(username: str, passwd_hash: str, email: str | None = None) -> bool:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (user, passwd, email) VALUES (?, ?, ?)",
                (username, passwd_hash, email),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def update_password(username: str, passwd_hash: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE users SET passwd = ? WHERE user = ?", (passwd_hash, username))
        return cur.rowcount > 0


def delete_user(username: str) -> Optional[str]:
    with _connect() as conn:
        cur = conn.execute("SELECT profile_pic FROM users WHERE user = ?", (username,))
        row = cur.fetchone()
        conn.execute("DELETE FROM users WHERE user = ?", (username,))
        return row[0] if row else None


def update_profile_pic(username: str, filename: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE users SET profile_pic = ? WHERE user = ?", (filename, username))
        return cur.rowcount > 0


def consume_daily_quota(username: str, daily_limit: int) -> Tuple[bool, int]:
    now = int(time.time())
    with _connect() as conn:
        cur = conn.execute(
            "SELECT daily_used, daily_reset_at FROM users WHERE user = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return False, 0
        used = row[0] or 0
        reset_at = row[1] or 0
        if now - reset_at >= 86_400:
            used = 0
            reset_at = now
        if used >= daily_limit:
            return False, 0
        used += 1
        conn.execute(
            "UPDATE users SET daily_used = ?, daily_reset_at = ? WHERE user = ?",
            (used, reset_at, username),
        )
        return True, max(daily_limit - used, 0)


def get_daily_usage(username: str, daily_limit: int) -> Tuple[int, int, int]:
    now = int(time.time())
    with _connect() as conn:
        cur = conn.execute(
            "SELECT daily_used, daily_reset_at FROM users WHERE user = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return 0, daily_limit, now
        used = row[0] or 0
        reset_at = row[1] or 0
        if now - reset_at >= 86_400:
            used = 0
            reset_at = now
            conn.execute(
                "UPDATE users SET daily_used = ?, daily_reset_at = ? WHERE user = ?",
                (used, reset_at, username),
            )
        remaining = max(daily_limit - used, 0)
        return used, remaining, reset_at
