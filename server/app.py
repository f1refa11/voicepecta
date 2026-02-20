import io
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageOps

from server.auth import create_token, decode_token, hash_password, verify_password
from server.config import AVATAR_DIR, TEMP_DIR, TRIAL_DURATION_HOURS, WEEKLY_TOKEN_LIMIT
from server.db import (
    create_user,
    consume_tokens,
    delete_user,
    get_user,
    get_user_by_email,
    get_user_limits,
    init_db,
    update_password,
    update_profile_pic,
)
from server.jobs import Job, JobStore
from server.transcribe import estimate_tokens_for_audio, transcribe_audio


try:
    RESAMPLE = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # Pillow<9 fallback
    RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]


app = FastAPI()
jobs = JobStore()


def _json_status(status: str, **extra) -> JSONResponse:
    payload = {"status": status}
    payload.update(extra)
    return JSONResponse(payload)


def _sse_event(event: str, data: str) -> bytes:
    lines = str(data).splitlines() or [""]
    body = f"event: {event}\n" + "\n".join(f"data: {line}" for line in lines) + "\n\n"
    return body.encode("utf-8")


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


async def _read_form_or_json(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return await request.json()
    form = await request.form()
    return dict(form)


def _get_username_from_token(token: Optional[str], authorization: Optional[str]) -> Optional[str]:
    token_value = token or _extract_bearer_token(authorization)
    if not token_value:
        return None
    return decode_token(token_value)


def _build_account_payload(username: str, row: dict, limits: Optional[dict] = None) -> dict:
    state = limits or get_user_limits(username) or {}
    return {
        "user": username,
        "profile_pic": row.get("profile_pic"),
        "plan": "trial",
        "trial_duration_hours": TRIAL_DURATION_HOURS,
        "trial_started_at": state.get("trial_started_at"),
        "trial_expires_at": state.get("trial_expires_at"),
        "trial_expired": bool(state.get("trial_expired", False)),
        "token_balance": int(state.get("token_balance", WEEKLY_TOKEN_LIMIT) or 0),
        "token_weekly_limit": WEEKLY_TOKEN_LIMIT,
        "token_reset_at": state.get("token_next_reset_at"),
        "token_next_reset_at": state.get("token_next_reset_at"),
    }


def _trial_expired_response(username: str, row: dict, limits: Optional[dict] = None) -> Optional[JSONResponse]:
    state = limits or get_user_limits(username)
    if not state or not state.get("trial_expired"):
        return None
    payload = _build_account_payload(username, row, state)
    payload["reason"] = "trial_expired"
    return _json_status("fail", **payload)


def _sse_error_response(message: str) -> StreamingResponse:
    return StreamingResponse(
        iter([_sse_event("error", message)]),
        media_type="text/event-stream",
    )


def _start_transcription_job(job: Job) -> None:
    def _run() -> None:
        try:
            if job.cancel_event.is_set():
                raise RuntimeError("cancelled")
            job.emit("status", "loading model")
            if job.cancel_event.is_set():
                raise RuntimeError("cancelled")
            job.emit("status", "transcribing")
            job.result = transcribe_audio(job.engine, job.model, job.audio_path, job.language, job.cancel_event)
            if job.cancel_event.is_set():
                raise RuntimeError("cancelled")
            job.emit("result", job.result)
            job.emit("status", "done")
        except Exception as exc:
            job.error = str(exc)
            job.emit("error", job.error)
        finally:
            job.close()
            jobs.remove(job.id)
            try:
                os.remove(job.audio_path)
            except OSError:
                pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


@app.on_event("startup")
def _on_startup() -> None:
    init_db()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/v1/auth/register")
async def register(request: Request) -> JSONResponse:
    payload = await _read_form_or_json(request)
    username = str(payload.get("login", "")).strip()
    password = str(payload.get("password", "")).strip()
    email = str(payload.get("email", "")).strip() or None
    if not username or not password:
        return _json_status("fail")
    if get_user(username):
        return _json_status("fail")
    if email and get_user_by_email(email):
        return _json_status("fail")
    created = create_user(username, hash_password(password), email=email)
    return _json_status("ok" if created else "fail")


@app.post("/v1/auth/login")
async def login(request: Request) -> JSONResponse:
    payload = await _read_form_or_json(request)
    username = str(payload.get("login", "")).strip()
    password = str(payload.get("password", "")).strip()
    if not username or not password:
        return _json_status("fail")
    row = get_user(username)
    if not row:
        return _json_status("fail")
    if not verify_password(password, str(row.get("passwd", ""))):
        return _json_status("fail")
    limits = get_user_limits(username)
    if not limits:
        return _json_status("fail")
    token = create_token(username)
    account_payload = _build_account_payload(username, row, limits)
    return _json_status(
        "ok",
        token=token,
        **account_payload,
    )


@app.post("/v1/account/changepasswd")
async def change_password(
    token: Optional[str] = Form(default=None),
    old_pass: str = Form(...),
    new_pass: str = Form(...),
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    username = _get_username_from_token(token, authorization)
    if not username or not old_pass or not new_pass:
        return _json_status("fail")
    row = get_user(username)
    if not row or not verify_password(old_pass, str(row.get("passwd", ""))):
        return _json_status("fail")
    trial_block = _trial_expired_response(username, row)
    if trial_block is not None:
        return trial_block
    new_hash = hash_password(new_pass)
    if not update_password(username, new_hash):
        return _json_status("fail")
    new_token = create_token(username)
    return _json_status("ok", token=new_token)


@app.post("/v1/account/destroy")
async def destroy_account(
    token: Optional[str] = Form(default=None),
    passwd: str = Form(...),
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    username = _get_username_from_token(token, authorization)
    if not username or not passwd:
        return _json_status("fail")
    row = get_user(username)
    if not row or not verify_password(passwd, str(row.get("passwd", ""))):
        return _json_status("fail")
    old_avatar = delete_user(username)
    if old_avatar:
        try:
            (AVATAR_DIR / old_avatar).unlink(missing_ok=True)
        except OSError:
            pass
    return _json_status("ok")


@app.post("/v1/account/profile_pic")
async def upload_profile_picture(
    request: Request,
    token: Optional[str] = Form(default=None),
    profile_pic: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    username = _get_username_from_token(token, authorization)
    if not username:
        return _json_status("fail")
    user = get_user(username)
    if not user:
        return _json_status("fail")
    trial_block = _trial_expired_response(username, user)
    if trial_block is not None:
        return trial_block
    try:
        content = await profile_pic.read()
        image = Image.open(io.BytesIO(content)).convert("RGBA")
        image = ImageOps.fit(image, (256, 256), method=RESAMPLE)
    except Exception:
        return _json_status("fail")
    filename = f"{uuid.uuid4().hex}.png"
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AVATAR_DIR / filename
    try:
        image.save(out_path, format="PNG")
    except Exception:
        return _json_status("fail")
    old_avatar = user.get("profile_pic")
    if not update_profile_pic(username, filename):
        out_path.unlink(missing_ok=True)
        return _json_status("fail")
    if old_avatar:
        try:
            (AVATAR_DIR / old_avatar).unlink(missing_ok=True)
        except OSError:
            pass
    return _json_status("ok", filename=filename)


@app.get("/avatars/{filename}")
async def get_avatar(filename: str) -> FileResponse:
    safe_name = os.path.basename(filename)
    path = AVATAR_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/v1/transcribe/{job_id}/cancel")
async def cancel_transcription(
    job_id: str,
    token: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    username = _get_username_from_token(token, authorization)
    if not username:
        return _json_status("fail")
    job = jobs.get(job_id)
    if not job or job.username != username:
        return _json_status("fail")
    job.cancel()
    jobs.remove(job_id)
    try:
        os.remove(job.audio_path)
    except OSError:
        pass
    return _json_status("ok")


@app.post("/v1/status")
async def status(request: Request, authorization: Optional[str] = Header(default=None)) -> JSONResponse:
    payload = await _read_form_or_json(request)
    token = payload.get("token")
    username = _get_username_from_token(token, authorization)
    if not username:
        return _json_status("fail")
    row = get_user(username)
    if not row:
        return _json_status("fail")
    limits = get_user_limits(username)
    if not limits:
        return _json_status("fail")
    return _json_status("ok", **_build_account_payload(username, row, limits))


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    engine: str = Form(...),
    model: str = Form(...),
    language: str | None = Form(default=None),
    token: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
) -> StreamingResponse:
    username = _get_username_from_token(token, authorization)
    row = get_user(username) if username else None
    if not username or not row:
        return _sse_error_response("auth failed")

    limits = get_user_limits(username)
    if not limits:
        return _sse_error_response("auth failed")
    if limits.get("trial_expired"):
        return _sse_error_response("trial expired")

    suffix = Path(file.filename or "").suffix
    temp_name = f"{os.urandom(8).hex()}{suffix}"
    temp_path = TEMP_DIR / temp_name
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        required_tokens = estimate_tokens_for_audio(engine, model, str(temp_path))
    except Exception as exc:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return _sse_error_response(f"token estimate failed: {exc}")

    ok, balance, _reset_at = consume_tokens(username, required_tokens)
    if not ok:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        missing = max(required_tokens - balance, 0)
        return _sse_error_response(
            f"insufficient tokens: balance={balance}, required={required_tokens}, missing={missing}"
        )

    job = Job(str(temp_path), engine, model, username, language)
    jobs.add(job)
    _start_transcription_job(job)

    def event_stream():
        queue = job.subscribe()
        yield _sse_event("job_id", job.id)
        while True:
            item = queue.get()
            if item is None:
                break
            event, data = item
            yield _sse_event(event, data)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/transcribe/{job_id}")
async def transcribe_status(
    job_id: str,
    token: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
) -> StreamingResponse:
    username = _get_username_from_token(token, authorization)
    if not username or not get_user(username):
        return StreamingResponse(
            iter([_sse_event("error", "auth failed")]),
            media_type="text/event-stream",
        )

    job = jobs.get(job_id)
    if not job:
        return StreamingResponse(
            iter([_sse_event("error", "job not found")]),
            media_type="text/event-stream",
        )

    def event_stream():
        queue = job.subscribe()
        yield _sse_event("job_id", job.id)
        while True:
            item = queue.get()
            if item is None:
                break
            event, data = item
            yield _sse_event(event, data)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
