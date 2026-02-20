import os
import threading
from typing import Any, Dict, Optional
import math

import numpy as np
import torch
import whisper
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

from server.config import VOSK_MODEL_DIR, WHISPER_SEGMENTS_WITH_TIMESTAMPS


_whisper_cache: Dict[str, Any] = {}
_vosk_cache: Dict[str, VoskModel] = {}

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


def estimate_tokens_for_audio(engine: str, model: str, audio_path: str) -> int:
    rate = get_tokens_per_second(engine, model)
    audio = whisper.load_audio(audio_path)
    duration_seconds = len(audio) / whisper.audio.SAMPLE_RATE
    return int(math.ceil(duration_seconds * rate))


def _get_whisper_model(model_name: str):
    if model_name not in _whisper_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_cache[model_name] = whisper.load_model(model_name, device=device)
    return _whisper_cache[model_name]


def _get_vosk_model(model_name: str) -> VoskModel:
    if model_name not in _vosk_cache:
        model_path = VOSK_MODEL_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {model_path}")
        SetLogLevel(-1)
        _vosk_cache[model_name] = VoskModel(str(model_path))
    return _vosk_cache[model_name]


def transcribe_audio(engine: str, model: str, audio_path: str, language: str | None = None, cancel_event: Optional[threading.Event] = None) -> str:
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
        model_obj = _get_vosk_model(model)
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
