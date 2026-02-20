import os
import threading
from typing import Any, Dict, Optional

import numpy as np
import torch
import whisper
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

from server.config import VOSK_MODEL_DIR


_whisper_cache: Dict[str, object] = {}
_vosk_cache: Dict[str, VoskModel] = {}


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
        model_obj = _get_whisper_model(model)
        kwargs = {"verbose": False}
        if language:
            kwargs["language"] = language
        result = model_obj.transcribe(audio_path, **kwargs)
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
