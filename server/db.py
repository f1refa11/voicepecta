import sqlite3
import time
from contextlib import closing
from typing import Any, Optional, Tuple

from server.config import (
    DB_BACKEND,
    DB_PATH,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
    TOKEN_RESET_INTERVAL_SECONDS,
    TRIAL_DURATION_HOURS,
    WEEKLY_TOKEN_LIMIT,
)

try:
    import pymysql
    from pymysql.cursors import DictCursor
except Exception:  # pragma: no cover - optional dependency in sqlite mode
    pymysql = None
    DictCursor = None


def _is_mysql_backend() -> bool:
    return DB_BACKEND in {"mysql", "mariadb"}


def _connect():
    if _is_mysql_backend():
        if pymysql is None:
            raise RuntimeError("MySQL backend selected but 'pymysql' is not installed")
        return pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            autocommit=False,
            cursorclass=DictCursor,
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _placeholder() -> str:
    return "%s" if _is_mysql_backend() else "?"


def _is_integrity_error(exc: Exception) -> bool:
    if isinstance(exc, sqlite3.IntegrityError):
        return True
    if pymysql is not None and isinstance(exc, pymysql.IntegrityError):
        return True
    return False


def _row_to_dict(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    if isinstance(row, dict):
        return row
    return None


def _ensure_users_table(conn) -> None:
    if _is_mysql_backend():
        conn.cursor().execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user VARCHAR(255) PRIMARY KEY,
                passwd TEXT NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        return
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user TEXT PRIMARY KEY,
            passwd TEXT NOT NULL
        )
        """
    )


def _get_columns(conn) -> set[str]:
    if _is_mysql_backend():
        with conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM users")
            rows = cur.fetchall()
        return {str(row.get("Field")) for row in rows}
    cur = conn.execute("PRAGMA table_info(users)")
    return {str(row[1]) for row in cur.fetchall()}


def _ensure_columns(conn) -> None:
    columns = _get_columns(conn)

    def add_column(name: str, sql_type: str) -> None:
        if name in columns:
            return
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE users ADD COLUMN {name} {sql_type}")
        else:
            conn.execute(f"ALTER TABLE users ADD COLUMN {name} {sql_type}")
        columns.add(name)

    email_type = "VARCHAR(320)" if _is_mysql_backend() else "TEXT"
    profile_pic_type = "VARCHAR(255)" if _is_mysql_backend() else "TEXT"
    int_type = "BIGINT" if _is_mysql_backend() else "INTEGER"

    add_column("email", email_type)
    add_column("profile_pic", profile_pic_type)
    add_column("token_balance", f"{int_type} DEFAULT {WEEKLY_TOKEN_LIMIT}")
    add_column("token_reset_at", f"{int_type} DEFAULT 0")
    add_column("trial_started_at", f"{int_type} DEFAULT 0")
    add_column("trial_expires_at", int_type)

    if _is_mysql_backend():
        with conn.cursor() as cur:
            cur.execute("SHOW INDEX FROM users WHERE Key_name = %s", ("idx_users_email_unique",))
            if not cur.fetchone():
                cur.execute("CREATE UNIQUE INDEX idx_users_email_unique ON users(email)")
    else:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email) WHERE email IS NOT NULL"
        )

    now = int(time.time())
    p = _placeholder()
    if _is_mysql_backend():
        with conn.cursor() as cur:
            cur.execute(f"UPDATE users SET token_balance = {p} WHERE token_balance IS NULL", (WEEKLY_TOKEN_LIMIT,))
            cur.execute(f"UPDATE users SET token_reset_at = {p} WHERE token_reset_at IS NULL OR token_reset_at = 0", (now,))
            cur.execute(f"UPDATE users SET trial_started_at = {p} WHERE trial_started_at IS NULL OR trial_started_at = 0", (now,))
            if TRIAL_DURATION_HOURS > 0:
                default_expiry = now + TRIAL_DURATION_HOURS * 3600
                cur.execute(f"UPDATE users SET trial_expires_at = {p} WHERE trial_expires_at IS NULL", (default_expiry,))
    else:
        conn.execute(f"UPDATE users SET token_balance = {p} WHERE token_balance IS NULL", (WEEKLY_TOKEN_LIMIT,))
        conn.execute(f"UPDATE users SET token_reset_at = {p} WHERE token_reset_at IS NULL OR token_reset_at = 0", (now,))
        conn.execute(f"UPDATE users SET trial_started_at = {p} WHERE trial_started_at IS NULL OR trial_started_at = 0", (now,))
        if TRIAL_DURATION_HOURS > 0:
            default_expiry = now + TRIAL_DURATION_HOURS * 3600
            conn.execute(f"UPDATE users SET trial_expires_at = {p} WHERE trial_expires_at IS NULL", (default_expiry,))


def init_db() -> None:
    with closing(_connect()) as conn:
        _ensure_users_table(conn)
        _ensure_columns(conn)
        conn.commit()


def _select_user_by_field(field: str, value: str) -> Optional[dict[str, Any]]:
    p = _placeholder()
    with closing(_connect()) as conn:
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT user, passwd, email, profile_pic, token_balance, token_reset_at, trial_started_at, trial_expires_at "
                    f"FROM users WHERE {field} = {p}",
                    (value,),
                )
                row = cur.fetchone()
        else:
            cur = conn.execute(
                f"SELECT user, passwd, email, profile_pic, token_balance, token_reset_at, trial_started_at, trial_expires_at "
                f"FROM users WHERE {field} = {p}",
                (value,),
            )
            row = cur.fetchone()
    return _row_to_dict(row)


def get_user(username: str) -> Optional[dict[str, Any]]:
    return _select_user_by_field("user", username)


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    return _select_user_by_field("email", email)


def create_user(username: str, passwd_hash: str, email: str | None = None) -> bool:
    now = int(time.time())
    trial_expires_at = now + TRIAL_DURATION_HOURS * 3600 if TRIAL_DURATION_HOURS > 0 else None
    p = _placeholder()
    try:
        with closing(_connect()) as conn:
            if _is_mysql_backend():
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO users (user, passwd, email, token_balance, token_reset_at, trial_started_at, trial_expires_at) "
                        f"VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p})",
                        (username, passwd_hash, email, WEEKLY_TOKEN_LIMIT, now, now, trial_expires_at),
                    )
            else:
                conn.execute(
                    f"INSERT INTO users (user, passwd, email, token_balance, token_reset_at, trial_started_at, trial_expires_at) "
                    f"VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p})",
                    (username, passwd_hash, email, WEEKLY_TOKEN_LIMIT, now, now, trial_expires_at),
                )
            conn.commit()
        return True
    except Exception as exc:
        if _is_integrity_error(exc):
            return False
        raise


def update_password(username: str, passwd_hash: str) -> bool:
    p = _placeholder()
    with closing(_connect()) as conn:
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(f"UPDATE users SET passwd = {p} WHERE user = {p}", (passwd_hash, username))
                changed = cur.rowcount > 0
        else:
            cur = conn.execute(f"UPDATE users SET passwd = {p} WHERE user = {p}", (passwd_hash, username))
            changed = cur.rowcount > 0
        conn.commit()
        return changed


def delete_user(username: str) -> Optional[str]:
    p = _placeholder()
    with closing(_connect()) as conn:
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(f"SELECT profile_pic FROM users WHERE user = {p}", (username,))
                row = cur.fetchone()
                cur.execute(f"DELETE FROM users WHERE user = {p}", (username,))
        else:
            cur = conn.execute(f"SELECT profile_pic FROM users WHERE user = {p}", (username,))
            row = cur.fetchone()
            conn.execute(f"DELETE FROM users WHERE user = {p}", (username,))
        conn.commit()

    row_dict = _row_to_dict(row)
    return str(row_dict.get("profile_pic")) if row_dict and row_dict.get("profile_pic") else None


def update_profile_pic(username: str, filename: str) -> bool:
    p = _placeholder()
    with closing(_connect()) as conn:
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(f"UPDATE users SET profile_pic = {p} WHERE user = {p}", (filename, username))
                changed = cur.rowcount > 0
        else:
            cur = conn.execute(f"UPDATE users SET profile_pic = {p} WHERE user = {p}", (filename, username))
            changed = cur.rowcount > 0
        conn.commit()
        return changed


def _normalize_token_window(conn, username: str, token_balance: Any, token_reset_at: Any) -> Tuple[int, int]:
    now = int(time.time())
    balance = int(token_balance if token_balance is not None else WEEKLY_TOKEN_LIMIT)
    reset_at = int(token_reset_at if token_reset_at else 0)

    should_reset = reset_at <= 0 or (now - reset_at) >= TOKEN_RESET_INTERVAL_SECONDS
    if should_reset:
        balance = WEEKLY_TOKEN_LIMIT
        reset_at = now
        p = _placeholder()
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE users SET token_balance = {p}, token_reset_at = {p} WHERE user = {p}",
                    (balance, reset_at, username),
                )
        else:
            conn.execute(
                f"UPDATE users SET token_balance = {p}, token_reset_at = {p} WHERE user = {p}",
                (balance, reset_at, username),
            )

    return balance, reset_at


def get_user_limits(username: str) -> Optional[dict[str, Any]]:
    now = int(time.time())
    p = _placeholder()
    with closing(_connect()) as conn:
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT token_balance, token_reset_at, trial_started_at, trial_expires_at FROM users WHERE user = {p}",
                    (username,),
                )
                row = cur.fetchone()
        else:
            cur = conn.execute(
                f"SELECT token_balance, token_reset_at, trial_started_at, trial_expires_at FROM users WHERE user = {p}",
                (username,),
            )
            row = cur.fetchone()

        row_dict = _row_to_dict(row)
        if not row_dict:
            return None

        balance, reset_at = _normalize_token_window(
            conn,
            username,
            row_dict.get("token_balance"),
            row_dict.get("token_reset_at"),
        )

        trial_started = int(row_dict.get("trial_started_at") or now)
        if TRIAL_DURATION_HOURS <= 0:
            trial_expires = None
            trial_expired = False
        else:
            trial_expires = trial_started + TRIAL_DURATION_HOURS * 3600
            stored_expiry = row_dict.get("trial_expires_at")
            if int(stored_expiry or 0) != trial_expires:
                if _is_mysql_backend():
                    with conn.cursor() as cur:
                        cur.execute(
                            f"UPDATE users SET trial_expires_at = {p} WHERE user = {p}",
                            (trial_expires, username),
                        )
                else:
                    conn.execute(
                        f"UPDATE users SET trial_expires_at = {p} WHERE user = {p}",
                        (trial_expires, username),
                    )
            trial_expired = bool(trial_expires is not None and now >= trial_expires)

        conn.commit()
        return {
            "token_balance": balance,
            "token_next_reset_at": reset_at + TOKEN_RESET_INTERVAL_SECONDS,
            "trial_started_at": trial_started,
            "trial_expires_at": trial_expires,
            "trial_expired": trial_expired,
        }


def consume_tokens(username: str, required_tokens: int) -> Tuple[bool, int, int]:
    required = max(0, int(required_tokens))
    p = _placeholder()

    with closing(_connect()) as conn:
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT token_balance, token_reset_at FROM users WHERE user = {p} FOR UPDATE",
                    (username,),
                )
                row = cur.fetchone()
        else:
            cur = conn.execute(
                f"SELECT token_balance, token_reset_at FROM users WHERE user = {p}",
                (username,),
            )
            row = cur.fetchone()

        row_dict = _row_to_dict(row)
        if not row_dict:
            conn.rollback()
            return False, 0, int(time.time()) + TOKEN_RESET_INTERVAL_SECONDS

        balance, reset_at = _normalize_token_window(
            conn,
            username,
            row_dict.get("token_balance"),
            row_dict.get("token_reset_at"),
        )

        if balance < required:
            conn.commit()
            return False, balance, reset_at + TOKEN_RESET_INTERVAL_SECONDS

        new_balance = balance - required
        if _is_mysql_backend():
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE users SET token_balance = {p} WHERE user = {p}",
                    (new_balance, username),
                )
        else:
            conn.execute(
                f"UPDATE users SET token_balance = {p} WHERE user = {p}",
                (new_balance, username),
            )
        conn.commit()
        return True, new_balance, reset_at + TOKEN_RESET_INTERVAL_SECONDS
