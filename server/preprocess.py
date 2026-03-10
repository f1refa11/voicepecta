import math
import os
import tempfile
import threading
from typing import Any, Optional

import soundfile as sf


DEEP_NOISE_REMOVER_TOKENS_PER_MINUTE = 3
NOISE_REMOVER_TOKENS_PER_MINUTE = 2

DEFAULT_PREPROCESSING_CONFIG = {
    "deep_noise_remover": {
        "enabled": False,
        "attenuation_limit_db": 12,
        "post_filter": False,
        "post_filter_beta": 0.02,
    },
    "noise_remover": {
        "enabled": False,
    },
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_beta(value: Any, default: float = 0.02) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(0.0, min(0.05, parsed))
    return round(round(parsed / 0.01) * 0.01, 2)


def normalize_preprocessing_config(raw: Any) -> dict[str, Any]:
    deep_default = DEFAULT_PREPROCESSING_CONFIG["deep_noise_remover"]
    noise_default = DEFAULT_PREPROCESSING_CONFIG["noise_remover"]

    deep_raw: Any = {}
    noise_raw: Any = {}
    if isinstance(raw, dict):
        deep_raw = raw.get("deep_noise_remover")
        noise_raw = raw.get("noise_remover")
    if not isinstance(deep_raw, dict):
        deep_raw = {}
    if not isinstance(noise_raw, dict):
        noise_raw = {}

    return {
        "deep_noise_remover": {
            "enabled": _as_bool(deep_raw.get("enabled"), bool(deep_default["enabled"])),
            "attenuation_limit_db": _as_int(
                deep_raw.get("attenuation_limit_db"),
                int(deep_default["attenuation_limit_db"]),
                0,
                100,
            ),
            "post_filter": _as_bool(deep_raw.get("post_filter"), bool(deep_default["post_filter"])),
            "post_filter_beta": _as_beta(deep_raw.get("post_filter_beta"), float(deep_default["post_filter_beta"])),
        },
        "noise_remover": {
            "enabled": _as_bool(noise_raw.get("enabled"), bool(noise_default["enabled"])),
        },
    }


def is_preprocessing_enabled(preprocessing: dict[str, Any]) -> bool:
    config = normalize_preprocessing_config(preprocessing)
    deep_enabled = bool(config["deep_noise_remover"]["enabled"])
    noise_enabled = bool(config["noise_remover"]["enabled"])
    return deep_enabled or noise_enabled


def estimate_preprocessing_tokens(duration_seconds: float, preprocessing: dict[str, Any]) -> int:
    duration = max(0.0, float(duration_seconds))
    duration_minutes = duration / 60.0
    config = normalize_preprocessing_config(preprocessing)
    tokens = 0
    if config["deep_noise_remover"]["enabled"]:
        tokens += int(math.ceil(duration_minutes * DEEP_NOISE_REMOVER_TOKENS_PER_MINUTE))
    if config["noise_remover"]["enabled"]:
        tokens += int(math.ceil(duration_minutes * NOISE_REMOVER_TOKENS_PER_MINUTE))
    return tokens


def _make_temp_wav_path() -> str:
    fd, out_path = tempfile.mkstemp(prefix="voicepecta-pre-", suffix=".wav")
    os.close(fd)
    return out_path


def _apply_deep_noise_remover(
    input_path: str,
    output_path: str,
    settings: dict[str, Any],
    cancel_event: Optional[threading.Event] = None,
) -> None:
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("cancelled")

    os.environ["DEVICE"] = "cpu"

    try:
        from df.enhance import enhance, init_df  # type: ignore[import-not-found]
        from df.io import load_audio  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Deep Noise Remover requires DeepFilterNet (install package: deepfilternet)."
        ) from exc

    post_filter = bool(settings.get("post_filter"))
    atten_lim = float(settings.get("attenuation_limit_db", 12))
    post_filter_beta = _as_beta(settings.get("post_filter_beta", 0.02))
    print(atten_lim, post_filter_beta)

    try:
        model, df_state, *_unused = init_df(
            post_filter=post_filter,
            log_level="ERROR",
            log_file=None,
            config_allow_defaults=True,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Deep Noise Remover could not initialize DeepFilterNet with this version."
        ) from exc
    model = model.to("cpu")
    audio, _meta = load_audio(input_path, df_state.sr(), "cpu")
    enhanced = enhance(model, df_state, audio, pad=True, atten_lim_db=atten_lim)

    if post_filter and post_filter_beta > 0.0:
        enhanced = (1.0 - post_filter_beta) * enhanced + post_filter_beta * audio

    enhanced_np = enhanced.detach().cpu().numpy().T
    sf.write(output_path, enhanced_np, df_state.sr())
    sf.write("/home/firefall/test123.wav", enhanced_np, df_state.sr())

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("cancelled")


def _apply_noise_remover(
    input_path: str,
    output_path: str,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("cancelled")

    try:
        from pyrnnoise import RNNoise  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Noise Remover requires RNNoise support (install package: pyrnnoise)."
        ) from exc

    sample_rate = int(sf.info(input_path).samplerate or 48_000)
    denoiser = RNNoise(sample_rate=sample_rate)
    for _speech_prob in denoiser.denoise_wav(input_path, output_path):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("cancelled")


def preprocess_audio(
    audio_path: str,
    preprocessing: dict[str, Any],
    cancel_event: Optional[threading.Event] = None,
) -> tuple[str, list[str]]:
    config = normalize_preprocessing_config(preprocessing)
    generated_paths: list[str] = []
    current_path = audio_path

    if not is_preprocessing_enabled(config):
        return current_path, generated_paths

    try:
        deep_cfg = config["deep_noise_remover"]
        if deep_cfg["enabled"]:
            deep_out = _make_temp_wav_path()
            _apply_deep_noise_remover(current_path, deep_out, deep_cfg, cancel_event=cancel_event)
            generated_paths.append(deep_out)
            current_path = deep_out

        noise_cfg = config["noise_remover"]
        if noise_cfg["enabled"]:
            noise_out = _make_temp_wav_path()
            _apply_noise_remover(current_path, noise_out, cancel_event=cancel_event)
            generated_paths.append(noise_out)
            current_path = noise_out

        return current_path, generated_paths
    except Exception:
        for path in generated_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        raise
