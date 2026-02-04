import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from server.auth import create_token, decode_token, hash_password, verify_password
from server.config import TEMP_DIR
from server.db import create_user, get_user, init_db
from server.jobs import Job, JobStore
from server.transcribe import transcribe_audio


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


def _start_transcription_job(job: Job) -> None:
    def _run() -> None:
        try:
            job.emit("status", "loading model")
            job.emit("status", "transcribing")
            job.result = transcribe_audio(job.engine, job.model, job.audio_path, job.language)
            job.emit("result", job.result)
            job.emit("status", "done")
        except Exception as exc:
            job.error = str(exc)
            job.emit("error", job.error)
        finally:
            job.close()
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


@app.post("/v1/auth/register")
async def register(request: Request) -> JSONResponse:
    payload = await _read_form_or_json(request)
    username = str(payload.get("login", "")).strip()
    password = str(payload.get("password", "")).strip()
    if not username or not password:
        return _json_status("fail")
    if get_user(username):
        return _json_status("fail")
    created = create_user(username, hash_password(password))
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
    if not verify_password(password, row["passwd"]):
        return _json_status("fail")
    token = create_token(username)
    return _json_status("ok", token=token)


@app.post("/v1/status")
async def status(request: Request, authorization: Optional[str] = Header(default=None)) -> JSONResponse:
    payload = await _read_form_or_json(request)
    token = payload.get("token")
    username = _get_username_from_token(token, authorization)
    if not username or not get_user(username):
        return _json_status("fail")
    return _json_status("ok", user=username)


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
    if not username or not get_user(username):
        return StreamingResponse(
            iter([_sse_event("error", "auth failed")]),
            media_type="text/event-stream",
        )

    suffix = Path(file.filename or "").suffix
    temp_name = f"{os.urandom(8).hex()}{suffix}"
    temp_path = TEMP_DIR / temp_name
    content = await file.read()
    temp_path.write_bytes(content)

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
