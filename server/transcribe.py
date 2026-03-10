import logging
import math
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import whisper
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

from server.config import VOSK_MODEL_DIR, WHISPER_SEGMENTS_WITH_TIMESTAMPS


logger = logging.getLogger(__name__)

_whisper_cache: Dict[str, Any] = {}
_vosk_cache: Dict[str, VoskModel] = {}
_vosk_download_locks: Dict[str, threading.Lock] = {}

_VOSK_DOWNLOAD_TIMEOUT_SECONDS = 30
_VOSK_DOWNLOAD_MAX_RETRIES = 3
_VOSK_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_VOSK_DOWNLOAD_PROGRESS_LOG_INTERVAL_SECONDS = 5.0

_WHISPER_TOKENS_PER_SECOND = {
    "tiny": 1,
    "base": 2,
    "small": 3,
    "medium": 4,
    "turbo": 5,
    "large": 6,
}

_VOSK_TOKENS_PER_SECOND = {
    "vosk-model-ru-0.42": 3,
    "vosk-model-small-ru-0.22": 1,
    "vosk-model-ru-0.22": 3,
    "vosk-model-ru-0.10": 5,
    "base model": 3,
    "lite model": 1,
    "legacy base model": 3,
    "legacy large model": 5,
    "(legacy) base model": 3,
    "(legacy) large model": 5,
}

_VOSK_MODEL_URLS = {
    "vosk-model-ru-0.42": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
    "vosk-model-small-ru-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
    "vosk-model-ru-0.22": "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip",
    "vosk-model-ru-0.10": "https://alphacephei.com/vosk/models/vosk-model-ru-0.10.zip",
}

_VOSK_MODEL_ALIASES = {
    "base model": "vosk-model-ru-0.42",
    "lite model": "vosk-model-small-ru-0.22",
    "legacy base model": "vosk-model-ru-0.22",
    "legacy large model": "vosk-model-ru-0.10",
    "(legacy) base model": "vosk-model-ru-0.22",
    "(legacy) large model": "vosk-model-ru-0.10",
}


def _emit_status(status_callback: Optional[Callable[[str], None]], message: str) -> None:
    if status_callback is None:
        return
    try:
        status_callback(message)
    except Exception:
        logger.exception("Failed to emit status callback: %s", message)


def _format_segment_timestamp(seconds: Any) -> str | None:
    try:
        total_ms = max(0, int(round(float(seconds) * 1000)))
    except (TypeError, ValueError):
        return None
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_whisper_result(result: dict[str, Any]) -> str:
    if not WHISPER_SEGMENTS_WITH_TIMESTAMPS:
        return str(result.get("text", "")).strip()
    segments = result.get("segments")
    if isinstance(segments, list):
        lines: list[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start = _format_segment_timestamp(segment.get("start"))
            end = _format_segment_timestamp(segment.get("end"))
            if start and end:
                lines.append(f"[{start} --> {end}] {text}")
            else:
                lines.append(text)
        if lines:
            return "\n".join(lines)
    return str(result.get("text", "")).strip()


def get_tokens_per_second(engine: str, model: str) -> int:
    engine_lower = engine.strip().lower()
    model_lower = model.strip().lower()
    if engine_lower == "whisper":
        if model_lower not in _WHISPER_TOKENS_PER_SECOND:
            raise ValueError(f"Unsupported Whisper model for token rates: {model}")
        return _WHISPER_TOKENS_PER_SECOND[model_lower]
    if engine_lower == "vosk":
        if model_lower not in _VOSK_TOKENS_PER_SECOND:
            raise ValueError(f"Unsupported Vosk model for token rates: {model}")
        return _VOSK_TOKENS_PER_SECOND[model_lower]
    raise ValueError(f"Unsupported engine for token rates: {engine}")


def get_audio_duration_seconds(audio_path: str) -> float:
    audio = whisper.load_audio(audio_path)
    return len(audio) / whisper.audio.SAMPLE_RATE


def estimate_tokens_for_duration(engine: str, model: str, duration_seconds: float) -> int:
    rate = get_tokens_per_second(engine, model)
    return int(math.ceil(max(float(duration_seconds), 0.0) * rate))


def estimate_tokens_for_audio(engine: str, model: str, audio_path: str) -> int:
    duration_seconds = get_audio_duration_seconds(audio_path)
    return estimate_tokens_for_duration(engine, model, duration_seconds)


def _get_whisper_model(model_name: str):
    if model_name not in _whisper_cache:
        _whisper_cache[model_name] = whisper.load_model(model_name, device="cpu")
    return _whisper_cache[model_name]


def _resolve_vosk_model_folder(model_name: str) -> str:
    model_raw = model_name.strip()
    model_lower = model_name.strip().lower()
    if model_lower in _VOSK_MODEL_URLS:
        return model_lower
    alias = _VOSK_MODEL_ALIASES.get(model_lower)
    if alias is not None:
        return alias
    if (VOSK_MODEL_DIR / model_raw).exists():
        return model_raw
    if model_lower != model_raw and (VOSK_MODEL_DIR / model_lower).exists():
        return model_lower
    raise ValueError(f"Unsupported Vosk model: {model_name}")


def _is_vosk_model_ready(model_path: Path) -> bool:
    if not model_path.is_dir():
        return False
    marker_paths = [
        model_path / "am" / "final.mdl",
        model_path / "conf" / "model.conf",
        model_path / "graph",
    ]
    if any(path.exists() for path in marker_paths):
        return True
    try:
        return any(model_path.iterdir())
    except OSError:
        return False


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _download_vosk_archive(
    model_folder: str,
    url: str,
    zip_path: Path,
    cancel_event: Optional[threading.Event] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "VoicepectaServer/1.0"},
    )

    for attempt in range(1, _VOSK_DOWNLOAD_MAX_RETRIES + 1):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("cancelled")

        _remove_path(zip_path)
        _emit_status(status_callback, f"downloading vosk model ({attempt}/{_VOSK_DOWNLOAD_MAX_RETRIES})")
        logger.info(
            "Downloading Vosk model '%s' (attempt %s/%s) from %s",
            model_folder,
            attempt,
            _VOSK_DOWNLOAD_MAX_RETRIES,
            url,
        )

        downloaded = 0
        total_size: Optional[int] = None
        content_type = ""
        try:
            with urllib.request.urlopen(request, timeout=_VOSK_DOWNLOAD_TIMEOUT_SECONDS) as response, open(
                zip_path,
                "wb",
            ) as out_file:
                content_length_header = response.headers.get("Content-Length")
                if content_length_header:
                    try:
                        total_size = int(content_length_header)
                    except (TypeError, ValueError):
                        total_size = None
                content_type = response.headers.get("Content-Type", "")
                status_code = getattr(response, "status", "unknown")
                logger.info(
                    "Vosk download response for '%s': status=%s, content_type=%s, content_length=%s",
                    model_folder,
                    status_code,
                    content_type,
                    total_size if total_size is not None else "unknown",
                )

                last_progress_log = time.monotonic()
                while True:
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("cancelled")
                    chunk = response.read(_VOSK_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    if now - last_progress_log >= _VOSK_DOWNLOAD_PROGRESS_LOG_INTERVAL_SECONDS:
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            logger.info(
                                "Vosk model '%s' download progress: %.1f%% (%s/%s bytes)",
                                model_folder,
                                progress,
                                downloaded,
                                total_size,
                            )
                        else:
                            logger.info(
                                "Vosk model '%s' download progress: %s bytes",
                                model_folder,
                                downloaded,
                            )
                        last_progress_log = now

                out_file.flush()
                os.fsync(out_file.fileno())

            if downloaded <= 0:
                raise RuntimeError(
                    f"Received 0 bytes while downloading Vosk model '{model_folder}' from {url}"
                )
            if total_size is not None and downloaded < total_size:
                raise RuntimeError(
                    f"Incomplete download for Vosk model '{model_folder}': "
                    f"{downloaded}/{total_size} bytes"
                )

            logger.info(
                "Downloaded Vosk model '%s' archive to %s (%s bytes)",
                model_folder,
                zip_path,
                downloaded,
            )
            return
        except RuntimeError:
            _remove_path(zip_path)
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            _remove_path(zip_path)
            logger.warning(
                "Failed to download Vosk model '%s' on attempt %s/%s: %s",
                model_folder,
                attempt,
                _VOSK_DOWNLOAD_MAX_RETRIES,
                exc,
            )
            if attempt == _VOSK_DOWNLOAD_MAX_RETRIES:
                raise RuntimeError(
                    "Failed to download Vosk model "
                    f"'{model_folder}' after {_VOSK_DOWNLOAD_MAX_RETRIES} attempts. "
                    f"Last error: {exc}"
                ) from exc
            time.sleep(min(attempt, 5))


def _download_vosk_model(
    model_folder: str,
    cancel_event: Optional[threading.Event] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    model_path = VOSK_MODEL_DIR / model_folder
    if _is_vosk_model_ready(model_path):
        return

    if model_path.exists() and not model_path.is_dir():
        logger.warning("Removing invalid Vosk model path (expected directory): %s", model_path)
        _remove_path(model_path)

    lock = _vosk_download_locks.setdefault(model_folder, threading.Lock())
    if lock.locked():
        logger.info("Waiting for in-progress Vosk model download: %s", model_folder)
    with lock:
        if _is_vosk_model_ready(model_path):
            return
        if model_path.exists() and not model_path.is_dir():
            logger.warning("Removing invalid Vosk model path before download: %s", model_path)
            _remove_path(model_path)

        url = _VOSK_MODEL_URLS.get(model_folder)
        if not url:
            raise ValueError(f"No download URL configured for Vosk model: {model_folder}")

        VOSK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = VOSK_MODEL_DIR / f"{model_folder}.zip"
        try:
            _download_vosk_archive(
                model_folder,
                url,
                zip_path,
                cancel_event=cancel_event,
                status_callback=status_callback,
            )

            if not zipfile.is_zipfile(zip_path):
                size = zip_path.stat().st_size if zip_path.exists() else 0
                raise RuntimeError(
                    f"Downloaded Vosk archive is not a valid zip file: {zip_path} (size={size} bytes)"
                )

            _emit_status(status_callback, "extracting vosk model")
            logger.info("Extracting Vosk model '%s' from %s", model_folder, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(VOSK_MODEL_DIR)

            if not _is_vosk_model_ready(model_path):
                existing_dirs = sorted(
                    item.name for item in VOSK_MODEL_DIR.iterdir() if item.is_dir()
                )
                raise FileNotFoundError(
                    "Downloaded archive for "
                    f"{model_folder}, but expected folder was not found or is incomplete. "
                    f"Directories in {VOSK_MODEL_DIR}: {existing_dirs}"
                )
            logger.info("Vosk model '%s' is ready at %s", model_folder, model_path)
        except Exception:
            logger.exception("Failed to prepare Vosk model '%s'", model_folder)
            _remove_path(model_path)
            raise
        finally:
            _remove_path(zip_path)


def _get_vosk_model(
    model_name: str,
    cancel_event: Optional[threading.Event] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> VoskModel:
    model_folder = _resolve_vosk_model_folder(model_name)
    if model_folder not in _vosk_cache:
        model_path = VOSK_MODEL_DIR / model_folder
        if not _is_vosk_model_ready(model_path):
            _download_vosk_model(
                model_folder,
                cancel_event=cancel_event,
                status_callback=status_callback,
            )
        _emit_status(status_callback, "loading vosk model")
        logger.info("Loading Vosk model '%s' from %s", model_folder, model_path)
        SetLogLevel(-1)
        _vosk_cache[model_folder] = VoskModel(str(model_path))
    return _vosk_cache[model_folder]


def transcribe_audio(
    engine: str,
    model: str,
    audio_path: str,
    language: str | None = None,
    cancel_event: Optional[threading.Event] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    engine_lower = engine.strip().lower()
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("cancelled")
    if engine_lower == "whisper":
        model_obj: Any = _get_whisper_model(model)
        if language:
            result = model_obj.transcribe(audio_path, language=language, verbose=False)
        else:
            result = model_obj.transcribe(audio_path, verbose=False)
        return _format_whisper_result(result)
    if engine_lower == "vosk":
        model_obj = _get_vosk_model(
            model,
            cancel_event=cancel_event,
            status_callback=status_callback,
        )
        audio = whisper.load_audio(audio_path)
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        recognizer = KaldiRecognizer(model_obj, whisper.audio.SAMPLE_RATE)
        chunk_size = 4000
        total_samples = len(audio_int16)
        for i in range(0, total_samples, chunk_size):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("cancelled")
            chunk = audio_int16[i : i + chunk_size]
            recognizer.AcceptWaveform(chunk.tobytes())
        final_result = recognizer.FinalResult()
        try:
            import json

            payload = json.loads(final_result)
            return str(payload.get("text", "")).strip()
        except Exception:
            return str(final_result).strip()
    raise ValueError("Unsupported engine")
