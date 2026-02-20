import tkinter
import tkinter
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image, ImageDraw, ImageOps
import logging
import threading
import sys
import io
import re
import os
import json
import urllib.request
import urllib.error
import urllib.parse
import zipfile
import numpy as np
from typing import Any, Optional
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import tempfile
import uuid

log = logging.getLogger(__name__)

# --- Globals ---
audio_file_path = None
recording_stream = None
recording_chunks = []
is_recording = False
is_paused = False
recording_start_time: Optional[datetime] = None
recording_timer_job = None
prepared_profile_pic_path: Optional[str] = None
profile_preview_image = None
profile_preview_path: Optional[str] = None
remote_transcribe_job_id: Optional[str] = None
remote_transcribe_active = False
account_dialog: Optional[customtkinter.CTkToplevel] = None
selected_mic_index = None
VERSION = "0.0.4"
current_ui_language = "ru"

# --- Internationalization ---
i18n = {
    "en": {
        "title": f"Voicepecta v{VERSION}",
        "select_audio_file": "Select Audio File",
        "no_file_selected": "No file selected",
        "whisper_model": "Whisper Model",
        "transcribe": "Transcribe",
        "transcribing": "Transcribing...",
        "settings": "Settings",
        "select_file_prompt": "Please select an audio file first.",
        "error_transcription": "An error occurred: {e}",
        "settings_title": "Options",
        "ui_language": "UI Language",
        "theme": "Appearance",
        "downloading_model": "Downloading model...",
        "downloading_model_prefix": "Downloading model: ",
        "transcribing_prefix": "Transcribing: ",
        "loading_audio": "Loading audio...",
        "use_cpu": "Use CPU",
        "engine": "Transcription Engine",
        "vosk_model": "Vosk Model",
        "download_prompt_title": "Download model?",
        "download_prompt_body": "The model \"{model}\" is not downloaded yet.\nSize: {size}\nDo you want to download it now?",
        "download_cancelled": "Download cancelled by user.",
        "extracting_model": "Extracting model...",
        "unknown_size": "Unknown size",
        "record": "Record",
        "stop_recording": "Stop",
        "pause_recording": "Pause",
        "resume_recording": "Resume",
        "recording_status": "Recording...",
        "recording_paused": "Recording paused",
        "recording_saved": "Recording saved: {name}",
        "mic_device": "Microphone",
        "no_mic_devices": "No input devices found",
        "account_settings_dialog_title": "Account settings",
        "change_profile_picture": "Change profile picture",
        "choose_image": "Choose image",
        "save_profile_picture": "Save profile picture",
        "change_password": "Change password",
        "current_password": "Current password",
        "new_password": "New password",
        "confirm_new_password": "New password again",
        "save_password": "Save password",
        "delete_account": "Delete account",
        "delete_account_confirm": "Are you SURE that you want to delete your account? This action is irreversible!",
        "delete_account_checkbox": "I'm aware this action is irreversible and I would lose my account forever",
        "destroy_account": "Destroy account",
        "account_deleted": "The account has been deleted. Thanks for using Voicepecta!",
        "profile_pic_saved": "Profile picture updated",
        "password_saved": "Password updated",
        "password_mismatch": "New passwords do not match",
        "password_required": "Please fill all password fields",
        "image_required": "Please choose an image first",
        "image_invalid": "Could not process image",
        "delete_account_failed": "Account deletion failed: {reason}",
        "profile_pic_failed": "Profile picture update failed: {reason}",
        "password_change_failed": "Password change failed: {reason}",
        "quit_prompt_transcribing": "The transcribing job is still running. Are you sure you want to quit Voicepecta?",
    },
    "ru": {
        "title": f"Voicepecta v{VERSION}",
        "select_audio_file": "Выберите аудиофайл",
        "no_file_selected": "Файл не выбран",
        "whisper_model": "Модель Whisper",
        "transcribe": "Транскрибировать",
        "transcribing": "Транскрибация...",
        "settings": "Настройки",
        "select_file_prompt": "Пожалуйста, сначала выберите аудиофайл.",
        "error_transcription": "Произошла ошибка: {e}",
        "settings_title": "Настройки",
        "ui_language": "Язык интерфейса",
        "theme": "Оформление",
        "downloading_model": "Загрузка модели...",
        "downloading_model_prefix": "Загрузка модели: ",
        "transcribing_prefix": "Транскрибация: ",
        "loading_audio": "Загрузка аудио...",
        "use_cpu": "Использовать ЦП",
        "engine": "Движок транскрибации",
        "vosk_model": "Модель Vosk",
        "download_prompt_title": "Скачать модель?",
        "download_prompt_body": "Модель \"{model}\" еще не скачана.\nРазмер: {size}\nСкачать сейчас?",
        "download_cancelled": "Загрузка отменена.",
        "extracting_model": "Распаковка модели...",
        "unknown_size": "Неизвестный размер",
        "record": "Запись",
        "stop_recording": "Стоп",
        "pause_recording": "Пауза",
        "resume_recording": "Продолжить",
        "recording_status": "Идет запись...",
        "recording_paused": "Запись приостановлена",
        "recording_saved": "Запись сохранена: {name}",
        "mic_device": "Микрофон",
        "no_mic_devices": "Устройства ввода не найдены",
        "account_settings_dialog_title": "Настройки аккаунта",
        "change_profile_picture": "Сменить аватар",
        "choose_image": "Выбрать изображение",
        "save_profile_picture": "Сохранить аватар",
        "change_password": "Сменить пароль",
        "current_password": "Текущий пароль",
        "new_password": "Новый пароль",
        "confirm_new_password": "Повторите пароль",
        "save_password": "Сохранить пароль",
        "delete_account": "Удалить аккаунт",
        "delete_account_confirm": "Вы уверены, что хотите удалить аккаунт? Это действие необратимо!",
        "delete_account_checkbox": "Я понимаю, что удалю аккаунт навсегда",
        "destroy_account": "Удалить аккаунт",
        "account_deleted": "Аккаунт удален. Спасибо, что использовали Voicepecta!",
        "profile_pic_saved": "Аватар обновлен",
        "password_saved": "Пароль обновлен",
        "password_mismatch": "Новые пароли не совпадают",
        "password_required": "Заполните все поля пароля",
        "image_required": "Сначала выберите изображение",
        "image_invalid": "Не удалось обработать изображение",
        "delete_account_failed": "Не удалось удалить аккаунт: {reason}",
        "profile_pic_failed": "Не удалось обновить аватар: {reason}",
        "password_change_failed": "Не удалось сменить пароль: {reason}",
        "quit_prompt_transcribing": "Задача транскрибации еще выполняется. Вы уверены, что хотите выйти из Voicepecta?",
    }
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
ENGINE_OPTIONS = ["Whisper", "Vosk"]
VOSK_MODELS = {
    "base model": {
        "folder": "vosk-model-ru-0.42",
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
        "size": "1.8 GB",
    },
    "lite model": {
        "folder": "vosk-model-small-ru-0.22",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
        "size": "45 MB",
    },
    "(legacy) base model": {
        "folder": "vosk-model-ru-0.22",
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip",
        "size": "1.5 GB",
    },
    "(legacy) large model": {
        "folder": "vosk-model-ru-0.10",
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.10.zip",
        "size": "2.5 GB",
    },
}
VOSK_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "vosk")

# Override i18n with corrected translations and extra strings
i18n = {
    "en": {
        "title": f"Voicepecta v{VERSION}",
        "select_audio_file": "Select Audio File",
        "no_file_selected": "No file selected",
        "whisper_model": "Whisper Model",
        "transcribe": "Transcribe",
        "transcribing": "Transcribing...",
        "settings": "Settings",
        "select_file_prompt": "Please select an audio file first.",
        "error_transcription": "An error occurred: {e}",
        "settings_title": "Options",
        "ui_language": "UI Language",
        "theme": "Appearance",
        "downloading_model": "Downloading model...",
        "downloading_model_prefix": "Downloading model: ",
        "transcribing_prefix": "Transcribing: ",
        "loading_audio": "Loading audio...",
        "use_cpu": "Use CPU",
        "engine": "Transcription Engine",
        "vosk_model": "Vosk Model",
        "download_prompt_title": "Download model?",
        "download_prompt_body": "The model \"{model}\" is not downloaded yet.\nSize: {size}\nDo you want to download it now?",
        "download_cancelled": "Download cancelled by user.",
        "extracting_model": "Extracting model...",
        "unknown_size": "Unknown size",
        "record": "Record",
        "stop_recording": "Stop",
        "pause_recording": "Pause",
        "resume_recording": "Resume",
        "recording_status": "Recording...",
        "recording_paused": "Recording paused",
        "recording_saved": "Recording saved: {name}",
        "mic_device": "Microphone",
        "no_mic_devices": "No input devices found",
        "language_label": "Language",
        "account": "Account",
        "account_settings": "Account settings",
        "logout": "Log out",
        "logged_in_as": "logged in as:",
        "not_logged_in": "not logged in",
        "plan_placeholder": "plan: free trial (30 days left)",
        "limit_placeholder": "daily limit: {used} / {limit} transcriptions used (resets daily)",
        "login_dialog_title": "log in to the Voicepecta server",
        "register_dialog_title": "register to Voicepecta",
        "login_label": "login:",
        "password_label": "password:",
        "enter_login_password": "enter login and password",
        "contacting_server": "contacting server...",
        "login_successful": "login successful",
        "login_failed": "login failed: {reason}",
        "registration_success": "registration complete. please log in.",
        "registration_failed": "registration failed: {reason}",
        "use_local_version": "use local version",
        "quit_prompt": "do you want to quit Voicepecta?",
        "models_not_installed": "vosk and openai-whisper models are not installed. install them and try again",
        "register_link": "register to Voicepecta",
        "login_link": "login to Voicepecta",
        "login_button": "log in",
        "register_button": "register",
        "login_first": "Please log in to the Voicepecta server first.",
        "account_settings_dialog_title": "Account settings",
        "change_profile_picture": "Change profile picture",
        "choose_image": "Choose image",
        "save_profile_picture": "Save profile picture",
        "change_password": "Change password",
        "current_password": "Current password",
        "new_password": "New password",
        "confirm_new_password": "New password again",
        "save_password": "Save password",
        "delete_account": "Delete account",
        "delete_account_confirm": "Are you SURE that you want to delete your account? This action is irreversible!",
        "delete_account_checkbox": "I'm aware this action is irreversible and I would lose my account forever",
        "destroy_account": "Destroy account",
        "account_deleted": "The account has been deleted. Thanks for using Voicepecta!",
        "profile_pic_saved": "Profile picture updated",
        "password_saved": "Password updated",
        "password_mismatch": "New passwords do not match",
        "password_required": "Please fill all password fields",
        "image_required": "Please choose an image first",
        "image_invalid": "Could not process image",
        "delete_account_failed": "Account deletion failed: {reason}",
        "profile_pic_failed": "Profile picture update failed: {reason}",
        "password_change_failed": "Password change failed: {reason}",
        "quit_prompt_transcribing": "The transcribing job is still running. Are you sure you want to quit Voicepecta?",
    },
    "ru": {
        "title": f"Voicepecta v{VERSION}",
        "select_audio_file": "Выберите аудиофайл",
        "no_file_selected": "Файл не выбран",
        "whisper_model": "Модель Whisper",
        "transcribe": "Транскрибировать",
        "transcribing": "Транскрибация...",
        "settings": "Настройки",
        "select_file_prompt": "Пожалуйста, сначала выберите аудиофайл.",
        "error_transcription": "Произошла ошибка: {e}",
        "settings_title": "Настройки",
        "ui_language": "Язык интерфейса",
        "theme": "Оформление",
        "downloading_model": "Загрузка модели...",
        "downloading_model_prefix": "Загрузка модели: ",
        "transcribing_prefix": "Транскрибция: ",
        "loading_audio": "Загрузка аудио...",
        "use_cpu": "Использовать ЦП",
        "engine": "Движок транскрибации",
        "vosk_model": "Модель Vosk",
        "download_prompt_title": "Скачать модель?",
        "download_prompt_body": "Модель \"{model}\" еще не скачана.\nРазмер: {size}\nСкачать сейчас?",
        "download_cancelled": "Загрузка отменена.",
        "extracting_model": "Распаковка модели...",
        "unknown_size": "Неизвестный размер",
        "record": "Запись",
        "stop_recording": "Стоп",
        "pause_recording": "Пауза",
        "resume_recording": "Продолжить",
        "recording_status": "Идет запись...",
        "recording_paused": "Запись приостановлена",
        "recording_saved": "Запись сохранена: {name}",
        "mic_device": "Микрофон",
        "no_mic_devices": "Устройства ввода не найдены",
        "language_label": "Язык распознавания",
        "account": "Аккаунт",
        "account_settings": "Настройки аккаунта",
        "logout": "Выйти",
        "logged_in_as": "Вы вошли как:",
        "not_logged_in": "Не авторизован",
        "plan_placeholder": "Тариф: пробный период (осталось 30 дней)",
        "limit_placeholder": "Дневной лимит: {used} / {limit} транскрипций использовано (сбрасывается ежедневно)",
        "login_dialog_title": "Вход на сервер Voicepecta",
        "register_dialog_title": "Регистрация в Voicepecta",
        "login_label": "Логин:",
        "password_label": "Пароль:",
        "enter_login_password": "Введите логин и пароль",
        "contacting_server": "Соединение с сервером...",
        "login_successful": "Вход выполнен",
        "login_failed": "Не удалось войти: {reason}",
        "registration_success": "Регистрация завершена. войдите в систему.",
        "registration_failed": "Не удалось зарегистрироваться: {reason}",
        "use_local_version": "Использовать локальную версию",
        "quit_prompt": "Выйти из Voicepecta?",
        "models_not_installed": "Модели vosk и openai-whisper не установлены. Установите их и попробуйте снова.",
        "register_link": "Регистрация в Voicepecta",
        "login_link": "Войти в Voicepecta",
        "login_button": "Войти",
        "register_button": "Зарегистрироваться",
        "login_first": "Сначала войдите в сервер Voicepecta.",
        "account_settings_dialog_title": "Настройки аккаунта",
        "change_profile_picture": "Сменить аватар",
        "choose_image": "Выбрать изображение",
        "save_profile_picture": "Сохранить аватар",
        "change_password": "Сменить пароль",
        "current_password": "Текущий пароль",
        "new_password": "Новый пароль",
        "confirm_new_password": "Повторите пароль",
        "save_password": "Сохранить пароль",
        "delete_account": "Удалить аккаунт",
        "delete_account_confirm": "Вы уверены, что хотите удалить аккаунт? Это действие необратимо!",
        "delete_account_checkbox": "Я понимаю, что удалю аккаунт навсегда",
        "destroy_account": "Удалить аккаунт",
        "account_deleted": "Аккаунт удален. Спасибо, что использовали Voicepecta!",
        "profile_pic_saved": "Аватар обновлен",
        "password_saved": "Пароль обновлен",
        "password_mismatch": "Новые пароли не совпадают",
        "password_required": "Заполните все поля пароля",
        "image_required": "Сначала выберите изображение",
        "image_invalid": "Не удалось обработать изображение",
        "delete_account_failed": "Не удалось удалить аккаунт: {reason}",
        "profile_pic_failed": "Не удалось обновить аватар: {reason}",
        "password_change_failed": "Не удалось сменить пароль: {reason}",
        "quit_prompt_transcribing": "Задача транскрибации еще выполняется. Вы уверены, что хотите выйти из Voicepecta?",
    }
}
# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DEFAULT_CONFIG = {
    "ui_language": current_ui_language,
    "theme": "Dark",
    "engine": "Whisper",
    "model": "base",
    "transcription_language": "Russian",
    "use_cpu": False,
    "selected_mic_index": None,
    "last_audio_file": None,
    "use_external_server": None,  # None means not chosen yet
    "auth_token": None,
    "username": None,
    "profile_pic": None,
    "server_url": "http://95.31.11.50:7788",
    "onboarded": False,
}
config_data = DEFAULT_CONFIG.copy()
startup_dialog_active = False
_whisper_module = None
_torch_module = None
_vosk_module = None


def get_whisper():
    global _whisper_module
    if _whisper_module is None:
        import whisper as _w
        _whisper_module = _w
    return _whisper_module


def get_torch():
    global _torch_module
    if _torch_module is None:
        import torch as _t
        _torch_module = _t
    return _torch_module


def get_vosk():
    global _vosk_module
    if _vosk_module is None:
        import vosk as _v
        _vosk_module = _v
    return _vosk_module


def load_config():
    global config_data, current_ui_language, selected_mic_index, audio_file_path
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                config_data.update(data)
    except Exception as e:
        log.debug(f"Using default config, could not load {CONFIG_PATH}: {e}")
    # migrate old default port to new default
    if config_data.get("server_url") == "http://95.31.11.50:8000":
        config_data["server_url"] = "http://95.31.11.50:7788"
    current_ui_language = config_data.get("ui_language", "ru")
    selected_mic_index = config_data.get("selected_mic_index")
    audio_file_path = config_data.get("last_audio_file")
    if audio_file_path and not os.path.exists(audio_file_path):
        audio_file_path = None


def get_language_code() -> Optional[str]:
    lang = config_data.get("transcription_language", "Russian")
    if lang == "Auto":
        return None
    if lang == "English":
        return "en"
    return "ru"


def format_segment_timestamp(seconds: Any) -> Optional[str]:
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


def format_whisper_result_text(result: dict[str, Any]) -> str:
    segments = result.get("segments")
    if isinstance(segments, list):
        lines = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start = format_segment_timestamp(segment.get("start"))
            end = format_segment_timestamp(segment.get("end"))
            if start and end:
                lines.append(f"[{start} --> {end}] {text}")
            else:
                lines.append(text)
        if lines:
            return "\n".join(lines)
    return str(result.get("text", "")).strip()


def save_config():
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to write config to {CONFIG_PATH}: {e}")


load_config()

# --- Progress Bar Handling ---
class ProgressIOWrapper(object):
    def __init__(self, original_stream, progress_bar, progress_label, mode="download", total_duration=None):
        self.original_stream = original_stream
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.mode = mode
        self.total_duration = total_duration
        self.tqdm_regex = re.compile(r"(\d+)%\|.*?\|\s*(.*?)(?:\[|$)")
        self.timestamp_regex = re.compile(r"\[.*?-->\s*(\d{2}:\d{2}\.\d{3})\]")

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()

        if not s.strip(): return

        if self.mode == "download":
            lines = s.replace('\r', '\n').split('\n')
            for line in lines:
                match = self.tqdm_regex.search(line)
                if match:
                    percent = int(match.group(1))
                    details = match.group(2).strip()
                    self.progress_bar.set(percent / 100)
                    prefix = i18n[current_ui_language].get("downloading_model_prefix", "Downloading model: ")
                    self.progress_label.configure(text=f"{prefix}{details}")
                    root.update_idletasks()

        elif self.mode == "transcribe" and self.total_duration:
            match = self.timestamp_regex.search(s)
            if match:
                time_str = match.group(1)
                current_seconds = self.time_str_to_seconds(time_str)
                progress = min(current_seconds / self.total_duration, 1.0)
                self.progress_bar.set(progress)
                prefix = i18n[current_ui_language].get("transcribing_prefix", "Transcribing: ")
                percent_str = f"{int(progress * 100)}%"
                self.progress_label.configure(text=f"{prefix}{percent_str}")
                root.update_idletasks()

    def flush(self):
        self.original_stream.flush()

    def time_str_to_seconds(self, time_str):
        parts = time_str.split(':')
        seconds = 0.0
        if len(parts) == 2:
            seconds += int(parts[0]) * 60
            seconds += float(parts[1])
        elif len(parts) == 3:
            seconds += int(parts[0]) * 3600
            seconds += int(parts[1]) * 60
            seconds += float(parts[2])
        return seconds

# --- Functions ---
def update_ui_language(lang_choice: str):
    global current_ui_language
    current_ui_language = "ru" if lang_choice == "Russian" else "en"
    config_data["ui_language"] = current_ui_language
    save_config()
    
    lang_dict = i18n[current_ui_language]
    root.title(lang_dict["title"])
    selectFileButton.configure(text=lang_dict["select_audio_file"])
    if not audio_file_path:
        selected_file_label.configure(text=lang_dict["no_file_selected"])
    update_model_label()
    engine_label.configure(text=lang_dict["engine"])
    recordButton.configure(text=lang_dict["stop_recording"] if is_recording else lang_dict["record"])
    pauseButton.configure(text=lang_dict["pause_recording"] if not is_paused else lang_dict["resume_recording"])
    transcribeButton.configure(text=lang_dict["transcribe"])
    settingsButton.configure(text=lang_dict["settings"])
    cpu_checkbox.configure(text=lang_dict["use_cpu"])

def apply_ttk_theme(mode: str):
    try:
        style = ttk.Style()
        themes = style.theme_names()
        if mode.lower() == "light" and "sun-valley-light" in themes:
            style.theme_use("sun-valley-light")
        elif mode.lower() == "dark" and "sun-valley-dark" in themes:
            style.theme_use("sun-valley-dark")
        elif "sun-valley-dark" in themes:
            style.theme_use("sun-valley-dark")
    except Exception as e:
        log.warning(f"Could not apply ttk theme: {e}")


def change_theme(new_theme: str):
    try:
        customtkinter.set_appearance_mode(new_theme)
    except Exception as e:
        log.warning(f"Could not change appearance mode to {new_theme}: {e}")
        return
    apply_ttk_theme(new_theme)
    config_data["theme"] = new_theme
    save_config()

def open_settings_window():
    settings_win = customtkinter.CTkToplevel(root)
    settings_win.title(i18n[current_ui_language]["settings_title"])
    settings_win.geometry("350x260")
    settings_win.transient(root)
    settings_win.bind("<Escape>", lambda _e: settings_win.destroy())

    def update_settings_ui(lang_choice: str):
        lang_code = "ru" if lang_choice == "Russian" else "en"
        update_ui_language(lang_choice)
        settings_win.title(i18n[lang_code]["settings_title"])
        lang_label.configure(text=i18n[lang_code]["ui_language"])
        theme_label.configure(text=i18n[lang_code]["theme"])
        mic_label.configure(text=i18n[lang_code]["mic_device"])

    lang_label = customtkinter.CTkLabel(settings_win, text=i18n[current_ui_language]["ui_language"])
    lang_label.pack(pady=(10, 5), padx=10)
    lang_combo = customtkinter.CTkComboBox(settings_win, values=["English", "Russian"], command=update_settings_ui)
    lang_combo.set("English" if current_ui_language == "en" else "Russian")
    lang_combo.pack(pady=5, padx=10, fill="x")

    theme_label = customtkinter.CTkLabel(settings_win, text=i18n[current_ui_language]["theme"])
    theme_label.pack(pady=(10, 5), padx=10)
    theme_segmented_button = customtkinter.CTkSegmentedButton(settings_win, values=["Light", "Dark", "System"], command=change_theme)
    theme_segmented_button.set(customtkinter.get_appearance_mode())
    theme_segmented_button.pack(pady=5, padx=10, fill="x")

    mic_label = customtkinter.CTkLabel(settings_win, text=i18n[current_ui_language]["mic_device"])
    mic_label.pack(pady=(10, 5), padx=10)
    mic_devices, mic_display = get_input_devices()
    mic_combo = customtkinter.CTkComboBox(settings_win, values=mic_display, command=update_selected_mic)
    if mic_display:
        mic_combo.set(mic_display[0] if selected_mic_index is None else get_mic_display_name(selected_mic_index, mic_devices, mic_display))
    else:
        mic_combo.set(i18n[current_ui_language]["no_mic_devices"])
        mic_combo.configure(state="disabled")
    mic_combo.pack(pady=5, padx=10, fill="x")
    
    settings_win.grab_set()

def select_audio_file():
    global audio_file_path
    audio_file_path = filedialog.askopenfilename()
    if audio_file_path:
        selected_file_label.configure(text=audio_file_path.split("/")[-1])
        config_data["last_audio_file"] = audio_file_path
        save_config()

def human_readable_size(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return i18n[current_ui_language]["unknown_size"]
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{int(num_bytes)} B"

def get_remote_file_size(url: str) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size else None
    except Exception:
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                content_range = response.headers.get("Content-Range")
                if content_range and "/" in content_range:
                    return int(content_range.split("/")[-1])
        except Exception:
            return None
    return None

def get_whisper_model_url(model_name: str) -> Optional[str]:
    return get_whisper()._MODELS.get(model_name)

def get_whisper_model_path(model_name: str) -> Optional[str]:
    url = get_whisper_model_url(model_name)
    if not url:
        return None
    download_root = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    os.makedirs(download_root, exist_ok=True)
    return os.path.join(download_root, os.path.basename(url))

def is_whisper_model_downloaded(model_name: str) -> bool:
    path = get_whisper_model_path(model_name)
    return path is not None and os.path.exists(path)

def get_whisper_model_size_text(model_name: str) -> str:
    url = get_whisper_model_url(model_name)
    if not url:
        return i18n[current_ui_language]["unknown_size"]
    size = get_remote_file_size(url)
    return human_readable_size(size)

def get_vosk_model_info(model_key: str) -> dict:
    return VOSK_MODELS.get(model_key, {})

def is_vosk_model_downloaded(model_key: str) -> bool:
    model_info = get_vosk_model_info(model_key)
    if not model_info:
        return False
    model_dir = os.path.join(VOSK_MODEL_DIR, model_info["folder"])
    return os.path.isdir(model_dir)

def prompt_model_download(model_name: str, size_text: str) -> bool:
    lang_dict = i18n[current_ui_language]
    return messagebox.askyesno(
        title=lang_dict["download_prompt_title"],
        message=lang_dict["download_prompt_body"].format(model=model_name, size=size_text)
    )

def download_file(url: str, dest_path: str):
    progress_bar.set(0)
    progress_label.configure(text=i18n[current_ui_language]["downloading_model"])
    root.update_idletasks()
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_size = int(total_size) if total_size else None
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.set(progress)
                    percent = int(progress * 100)
                    prefix = i18n[current_ui_language].get("downloading_model_prefix", "Downloading model: ")
                    progress_label.configure(text=f"{prefix}{percent}%")
                    root.update_idletasks()
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

def download_and_extract_vosk(model_key: str):
    model_info = get_vosk_model_info(model_key)
    if not model_info:
        raise ValueError("Unknown Vosk model")
    os.makedirs(VOSK_MODEL_DIR, exist_ok=True)
    zip_path = os.path.join(VOSK_MODEL_DIR, f"{model_info['folder']}.zip")
    download_file(model_info["url"], zip_path)
    progress_label.configure(text=i18n[current_ui_language]["extracting_model"])
    progress_bar.set(0)
    root.update_idletasks()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(VOSK_MODEL_DIR)
    os.remove(zip_path)

def update_model_label():
    engine = engineOptionMenu.get()
    if engine == "Vosk":
        model_label.configure(text=i18n[current_ui_language]["vosk_model"])
    else:
        model_label.configure(text=i18n[current_ui_language]["whisper_model"])

def update_model_options(engine_choice: str):
    config_data["engine"] = engine_choice
    if engine_choice == "Vosk":
        modelOptionMenu.configure(values=list(VOSK_MODELS.keys()))
        modelOptionMenu.set("base model")
        cpu_checkbox.configure(state="disabled")
        config_data["model"] = "base model"
    else:
        modelOptionMenu.configure(values=WHISPER_MODELS)
        modelOptionMenu.set("base")
        cpu_checkbox.configure(state="normal")
        config_data["model"] = "base"
    update_model_label()
    save_config()


def on_model_selected(choice: str):
    config_data["model"] = choice
    save_config()


def on_cpu_toggle():
    config_data["use_cpu"] = bool(cpu_checkbox.get())
    save_config()


def build_server_url(path: str) -> str:
    base = config_data.get("server_url") or DEFAULT_CONFIG["server_url"]
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def _post_json(path: str, payload: dict) -> dict:
    url = build_server_url(path)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def _post_form(path: str, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]] | None = None) -> dict:
    url = build_server_url(path)
    if files:
        boundary = "----voicepectaBoundary" + os.urandom(8).hex()
        body = io.BytesIO()
        for name, value in fields.items():
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
        for name, (filename, content, content_type) in files.items():
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            body.write(content)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode("utf-8"))
        data = body.getvalue()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    else:
        encoded = urllib.parse.urlencode(fields).encode("utf-8")
        data = encoded
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def validate_token(token: str) -> bool:
    try:
        resp = _post_json("/v1/status", {"token": token})
        return resp.get("status") == "ok"
    except Exception as e:
        log.error(f"Token validation failed: {e}")
        return False


def attempt_login_remote(login: str, password: str) -> tuple[bool, str]:
    try:
        resp = _post_json("/v1/auth/login", {"login": login, "password": password})
    except Exception as e:
        log.error(f"Login request failed: {e}")
        return False, "network error"
    if resp.get("status") == "ok" and resp.get("token"):
        config_data.update({
            "auth_token": resp.get("token"),
            "username": login,
            "use_external_server": True,
            "onboarded": True,
            "profile_pic": resp.get("profile_pic"),
        })
        save_config()
        return True, ""
    return False, "invalid credentials"


def attempt_register_remote(login: str, password: str) -> tuple[bool, str]:
    try:
        resp = _post_json("/v1/auth/register", {"login": login, "password": password})
    except Exception as e:
        log.error(f"Register request failed: {e}")
        return False, "network error"
    if resp.get("status") == "ok":
        return True, ""
    return False, "failed"


def change_password_remote(old_pass: str, new_pass: str) -> tuple[bool, str]:
    token = config_data.get("auth_token")
    if not token:
        return False, "no token"
    try:
        resp = _post_form(
            "/v1/account/changepasswd",
            {"old_pass": old_pass, "new_pass": new_pass, "token": token},
        )
    except Exception as e:
        log.error(f"Change password failed: {e}")
        return False, "network error"
    if resp.get("status") == "ok" and resp.get("token"):
        config_data["auth_token"] = resp.get("token")
        save_config()
        return True, ""
    return False, resp.get("reason", "fail")


def destroy_account_remote(password: str) -> tuple[bool, str]:
    token = config_data.get("auth_token")
    if not token:
        return False, "no token"
    try:
        resp = _post_form("/v1/account/destroy", {"passwd": password, "token": token})
    except Exception as e:
        log.error(f"Destroy account failed: {e}")
        return False, "network error"
    if resp.get("status") == "ok":
        return True, ""
    return False, resp.get("reason", "fail")


def upload_profile_picture_remote(file_path: str) -> tuple[bool, str, Optional[str]]:
    token = config_data.get("auth_token")
    if not token:
        return False, "no token", None
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        resp = _post_form(
            "/v1/account/profile_pic",
            {"token": token},
            {"profile_pic": (os.path.basename(file_path), content, "image/png")},
        )
    except Exception as e:
        log.error(f"Upload profile picture failed: {e}")
        return False, "network error", None
    if resp.get("status") == "ok":
        filename = resp.get("filename")
        if filename:
            config_data["profile_pic"] = filename
            save_config()
        return True, "", filename
    return False, resp.get("reason", "fail"), None


def cancel_remote_job(job_id: str) -> bool:
    token = config_data.get("auth_token")
    if not token:
        return False
    try:
        resp = _post_form(f"/v1/transcribe/{job_id}/cancel", {"token": token})
        return resp.get("status") == "ok"
    except Exception as e:
        log.error(f"Cancel remote job failed: {e}")
        return False


def prepare_profile_image(file_path: str) -> Optional[str]:
    resample = _get_resample()
    try:
        img = Image.open(file_path).convert("RGBA")
        img = ImageOps.fit(img, (256, 256), method=resample)
        mask = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, 256, 256), fill=255)
        img.putalpha(mask)
        temp_path = os.path.join(tempfile.gettempdir(), f"voicepecta-avatar-{uuid.uuid4().hex}.png")
        img.save(temp_path, format="PNG")
        return temp_path
    except Exception as e:
        log.error(f"Failed to prepare profile image: {e}")
        return None


def _get_resample():
    try:
        return Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:
        return Image.LANCZOS  # type: ignore[attr-defined]


def fetch_remote_avatar(filename: str, size: tuple[int, int] = (70, 70)):
    if not filename:
        return None
    url = build_server_url(f"/avatars/{filename}")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        resample = _get_resample()
        img = ImageOps.fit(img, size, method=resample)
        return customtkinter.CTkImage(img, size=size)
    except Exception as e:
        log.error(f"Failed to fetch avatar: {e}")
        return None


def close_account_dialog():
    global account_dialog
    if account_dialog is not None:
        try:
            account_dialog.destroy()
        except Exception:
            pass
    account_dialog = None


def check_local_dependencies() -> bool:
    try:
        import whisper  # noqa: F401
        from vosk import Model as _  # noqa: F401
    except Exception:
        return False
    return True


def show_account_dialog():
    global account_dialog
    close_account_dialog()
    dlg = customtkinter.CTkToplevel(root)
    account_dialog = dlg
    dlg.title(i18n[current_ui_language]["account"])
    dlg.geometry("420x360")
    dlg.transient(root)
    dlg.lift()
    dlg.update_idletasks()
    dlg.wait_visibility()
    dlg.grab_set()
    dlg.focus_force()
    def on_close(_e=None):
        close_account_dialog()
    dlg.bind("<Escape>", on_close)
    dlg.protocol("WM_DELETE_WINDOW", on_close)

    wrapper = customtkinter.CTkFrame(dlg, fg_color="transparent")
    wrapper.pack(fill="both", expand=True, padx=20, pady=20)

    header = customtkinter.CTkFrame(wrapper, fg_color="transparent")
    header.pack(fill="x", pady=(0, 12))

    avatar = customtkinter.CTkFrame(header, width=70, height=70, corner_radius=40, border_width=2, fg_color="transparent")
    avatar.pack(side="left", padx=(0, 12))
    avatar.pack_propagate(False)
    avatar_label = customtkinter.CTkLabel(avatar, text="")
    avatar_label.pack(expand=True)
    avatar_img = None
    profile_filename = config_data.get("profile_pic")
    if profile_filename:
        avatar_img = fetch_remote_avatar(profile_filename, size=(70, 70))
        if avatar_img:
            avatar_label.configure(image=avatar_img)
            avatar_label.image = avatar_img

    user_box = customtkinter.CTkFrame(header, fg_color="transparent")
    user_box.pack(side="left", fill="x", expand=True)
    customtkinter.CTkLabel(user_box, text=i18n[current_ui_language]["logged_in_as"], font=("Arial", 18)).pack(anchor="w", pady=(4, 0))
    username = config_data.get("username") or i18n[current_ui_language]["not_logged_in"]
    customtkinter.CTkLabel(user_box, text=username, font=("Arial", 20)).pack(anchor="w", pady=(0, 6))

    info_frame = customtkinter.CTkFrame(wrapper, fg_color="transparent")
    info_frame.pack(fill="x", pady=(0, 20))
    customtkinter.CTkLabel(info_frame, text=i18n[current_ui_language]["plan_placeholder"], font=("Arial", 18)).pack(anchor="w", pady=(0, 4))
    limit_label = customtkinter.CTkLabel(info_frame, text=i18n[current_ui_language]["limit_placeholder"].format(used=0, limit=5), font=("Arial", 18))
    limit_label.pack(anchor="w")

    def refresh_limit_label():
        token = config_data.get("auth_token")
        if not token:
            return
        try:
            resp = _post_json("/v1/status", {"token": token})
        except Exception as e:
            log.error(f"Status request failed: {e}")
            return
        if resp.get("status") != "ok":
            return
        used = int(resp.get("daily_used", 0) or 0)
        limit = int(resp.get("daily_limit", 5) or 5)
        limit_label.configure(text=i18n[current_ui_language]["limit_placeholder"].format(used=used, limit=limit))

    refresh_limit_label()

    def do_logout():
        config_data.update({
            "auth_token": None,
            "username": None,
            "use_external_server": None,
            "onboarded": False,
        })
        save_config()
        dlg.destroy()
        root.after(50, show_initial_dialog)

    customtkinter.CTkButton(wrapper, text=i18n[current_ui_language]["account_settings"], width=360, command=show_account_settings_dialog).pack(fill="x", pady=(0, 12))
    customtkinter.CTkButton(wrapper, text=i18n[current_ui_language]["logout"], width=360, command=do_logout).pack(fill="x")


def show_account_settings_dialog():
    global profile_preview_image, profile_preview_path, prepared_profile_pic_path
    dlg = customtkinter.CTkToplevel(root)
    dlg.title(i18n[current_ui_language]["account_settings_dialog_title"])
    dlg.geometry("540x560")
    dlg.transient(root)
    dlg.lift()
    dlg.update_idletasks()
    dlg.wait_visibility()
    dlg.grab_set()
    dlg.focus_force()
    dlg.bind("<Escape>", lambda _e: dlg.destroy())

    wrapper = customtkinter.CTkFrame(dlg, fg_color="transparent")
    wrapper.pack(fill="both", expand=True, padx=20, pady=20)

    customtkinter.CTkLabel(wrapper, text=i18n[current_ui_language]["account_settings_dialog_title"], font=("Arial", 20)).pack(anchor="w", pady=(0, 10))

    # Change profile picture
    customtkinter.CTkLabel(wrapper, text=i18n[current_ui_language]["change_profile_picture"], font=("Arial", 16)).pack(anchor="w")
    ttk.Separator(wrapper, orient="horizontal").pack(fill="x", pady=(4, 10))

    pic_row = customtkinter.CTkFrame(wrapper, fg_color="transparent")
    pic_row.pack(fill="x", pady=(0, 8))

    avatar_frame = customtkinter.CTkFrame(pic_row, width=90, height=90, corner_radius=45, fg_color="#e5e5e5")
    avatar_frame.pack(side="left", padx=(0, 12))
    avatar_frame.pack_propagate(False)
    preview_label = customtkinter.CTkLabel(avatar_frame, text="")
    preview_label.pack(expand=True)

    if profile_preview_path and os.path.exists(profile_preview_path):
        try:
            profile_preview_image = customtkinter.CTkImage(Image.open(profile_preview_path), size=(80, 80))
            preview_label.configure(image=profile_preview_image)
        except Exception:
            profile_preview_image = None

    button_column = customtkinter.CTkFrame(pic_row, fg_color="transparent")
    button_column.pack(side="left", fill="x", expand=True)

    def choose_image():
        nonlocal preview_label
        global profile_preview_path, prepared_profile_pic_path, profile_preview_image
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif"), ("All files", "*.*")])
        if not path:
            return
        processed = prepare_profile_image(path)
        if not processed:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["image_invalid"])
            return
        if profile_preview_path and profile_preview_path != processed:
            try:
                os.remove(profile_preview_path)
            except OSError:
                pass
        profile_preview_path = processed
        prepared_profile_pic_path = processed
        try:
            profile_preview_image = customtkinter.CTkImage(Image.open(processed), size=(80, 80))
            preview_label.configure(image=profile_preview_image)
        except Exception:
            profile_preview_image = None
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["image_invalid"])

    def save_profile_picture():
        if not prepared_profile_pic_path or not os.path.exists(prepared_profile_pic_path):
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["image_required"])
            return
        ok, reason, filename = upload_profile_picture_remote(prepared_profile_pic_path)
        if ok:
            messagebox.showinfo(i18n[current_ui_language]["title"], i18n[current_ui_language]["profile_pic_saved"])
        else:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["profile_pic_failed"].format(reason=reason))

    customtkinter.CTkButton(button_column, text=i18n[current_ui_language]["choose_image"], command=choose_image).pack(fill="x", pady=(0, 6))
    customtkinter.CTkButton(button_column, text=i18n[current_ui_language]["save_profile_picture"], command=save_profile_picture).pack(fill="x")

    # Change password
    customtkinter.CTkLabel(wrapper, text=i18n[current_ui_language]["change_password"], font=("Arial", 16)).pack(anchor="w", pady=(10, 0))
    ttk.Separator(wrapper, orient="horizontal").pack(fill="x", pady=(4, 10))

    pass_frame = customtkinter.CTkFrame(wrapper, fg_color="transparent")
    pass_frame.pack(fill="x", pady=(0, 8))

    def _add_pass_row(label_text):
        row = customtkinter.CTkFrame(pass_frame, fg_color="transparent")
        row.pack(fill="x", pady=(0, 6))
        customtkinter.CTkLabel(row, text=label_text, width=180, anchor="e").grid(row=0, column=0, padx=(0, 8))
        entry = customtkinter.CTkEntry(row, show="*")
        entry.grid(row=0, column=1, sticky="ew")
        row.grid_columnconfigure(1, weight=1)
        return entry

    current_pass_entry = _add_pass_row(i18n[current_ui_language]["current_password"])
    new_pass_entry = _add_pass_row(i18n[current_ui_language]["new_password"])
    confirm_pass_entry = _add_pass_row(i18n[current_ui_language]["confirm_new_password"])

    def save_password():
        old_pass = current_pass_entry.get().strip()
        new_pass = new_pass_entry.get().strip()
        confirm_pass = confirm_pass_entry.get().strip()
        if not old_pass or not new_pass or not confirm_pass:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["password_required"])
            return
        if new_pass != confirm_pass:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["password_mismatch"])
            return
        ok, reason = change_password_remote(old_pass, new_pass)
        if ok:
            messagebox.showinfo(i18n[current_ui_language]["title"], i18n[current_ui_language]["password_saved"])
            current_pass_entry.delete(0, tkinter.END)
            new_pass_entry.delete(0, tkinter.END)
            confirm_pass_entry.delete(0, tkinter.END)
        else:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["password_change_failed"].format(reason=reason))

    customtkinter.CTkButton(wrapper, text=i18n[current_ui_language]["save_password"], command=save_password).pack(fill="x", pady=(4, 10))

    # Delete account
    customtkinter.CTkLabel(wrapper, text=i18n[current_ui_language]["delete_account"], font=("Arial", 16)).pack(anchor="w", pady=(6, 0))
    ttk.Separator(wrapper, orient="horizontal").pack(fill="x", pady=(4, 10))

    def open_delete_confirm():
        confirm = customtkinter.CTkToplevel(dlg)
        confirm.title(i18n[current_ui_language]["delete_account"])
        confirm.geometry("420x220")
        confirm.transient(dlg)
        confirm.lift()
        confirm.update_idletasks()
        confirm.wait_visibility()
        confirm.grab_set()
        confirm.focus_force()
        confirm.bind("<Escape>", lambda _e: confirm.destroy())

        inner = customtkinter.CTkFrame(confirm, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=15)

        customtkinter.CTkLabel(inner, text=i18n[current_ui_language]["delete_account_confirm"], wraplength=360, justify="left").pack(anchor="w", pady=(0, 8))
        aware_var = tkinter.BooleanVar(value=False)
        customtkinter.CTkCheckBox(inner, text=i18n[current_ui_language]["delete_account_checkbox"], variable=aware_var).pack(anchor="w", pady=(0, 8))

        pass_row = customtkinter.CTkFrame(inner, fg_color="transparent")
        pass_row.pack(fill="x", pady=(0, 10))
        pass_entry = customtkinter.CTkEntry(pass_row, show="*")
        pass_entry.grid(row=0, column=0, sticky="ew")
        customtkinter.CTkLabel(pass_row, text=i18n[current_ui_language]["current_password"], width=180, anchor="w").grid(row=0, column=1, padx=(8, 0))
        pass_row.grid_columnconfigure(0, weight=1)

        def confirm_delete():
            if not aware_var.get():
                messagebox.showerror(
                    i18n[current_ui_language]["title"],
                    i18n[current_ui_language]["delete_account_failed"].format(reason="confirmation required"),
                )
                return
            passwd = pass_entry.get().strip()
            if not passwd:
                messagebox.showerror(
                    i18n[current_ui_language]["title"],
                    i18n[current_ui_language]["delete_account_failed"].format(reason="password required"),
                )
                return
            ok, reason = destroy_account_remote(passwd)
            if ok:
                messagebox.showinfo(i18n[current_ui_language]["title"], i18n[current_ui_language]["account_deleted"])
                config_data.update({
                    "auth_token": None,
                    "username": None,
                    "use_external_server": None,
                    "onboarded": False,
                    "profile_pic": None,
                })
                save_config()
                confirm.destroy()
                dlg.destroy()
                close_account_dialog()
                root.after(50, show_initial_dialog)
            else:
                messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["delete_account_failed"].format(reason=reason))

        customtkinter.CTkButton(inner, text=i18n[current_ui_language]["destroy_account"], fg_color="#d9534f", hover_color="#c9302c", command=confirm_delete).pack(fill="x")

    customtkinter.CTkButton(wrapper, text=i18n[current_ui_language]["delete_account"], fg_color="#d9534f", hover_color="#c9302c", command=open_delete_confirm).pack(fill="x")

def get_input_devices():
    devices = sd.query_devices()
    input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
    display = [f"{int(d['index'])}: {d['name']}" for d in input_devices]
    return input_devices, display

def get_mic_display_name(index: int, devices: list, display: list) -> str:
    for d, name in zip(devices, display):
        if int(d["index"]) == int(index):
            return name
    return display[0] if display else ""

def update_selected_mic(choice: str):
    global selected_mic_index
    try:
        selected_mic_index = int(choice.split(":")[0].strip())
    except Exception:
        selected_mic_index = None
    config_data["selected_mic_index"] = selected_mic_index
    save_config()

def _recording_callback(indata, frames, time_info, status):
    if status:
        log.warning(f"Recording status: {status}")
    if not is_recording or is_paused:
        return
    recording_chunks.append(indata.copy())

def _format_elapsed(delta) -> str:
    total_seconds = int(delta.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def _update_recording_timer():
    global recording_timer_job
    if not is_recording or recording_start_time is None:
        recording_timer_job = None
        return
    elapsed = datetime.now() - recording_start_time
    status = i18n[current_ui_language]["recording_paused"] if is_paused else i18n[current_ui_language]["recording_status"]
    progress_label.configure(text=f"{status} {_format_elapsed(elapsed)}")
    recording_timer_job = root.after(500, _update_recording_timer)

def _start_recording_timer():
    global recording_timer_job
    if recording_timer_job:
        root.after_cancel(recording_timer_job)
    _update_recording_timer()

def _stop_recording_timer():
    global recording_timer_job
    if recording_timer_job:
        root.after_cancel(recording_timer_job)
        recording_timer_job = None

def toggle_recording():
    if is_recording:
        stop_recording()
    else:
        start_recording()

def start_recording():
    global recording_stream, recording_chunks, is_recording, is_paused, audio_file_path, recording_start_time
    if is_recording:
        return
    recording_chunks = []
    is_recording = True
    is_paused = False
    recording_start_time = datetime.now()
    recordButton.configure(text=i18n[current_ui_language]["stop_recording"])
    pauseButton.configure(state="normal")
    progress_bar.configure(mode="indeterminate")
    progress_bar.start()
    progress_bar.pack(fill="x", padx=10, pady=(0, 4))
    progress_label.pack(pady=(0, 10), padx=10)
    _start_recording_timer()
    try:
        recording_stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            callback=_recording_callback,
            device=selected_mic_index
        )
        recording_stream.start()
    except Exception as e:
        log.error(f"Error starting recording: {e}")
        result_textbox.delete("1.0", tkinter.END)
        result_textbox.insert(tkinter.END, i18n[current_ui_language]["error_transcription"].format(e=e))
        stop_recording(save=False)

def toggle_pause():
    global is_paused
    if not is_recording:
        return
    is_paused = not is_paused
    if is_paused:
        pauseButton.configure(text=i18n[current_ui_language]["resume_recording"])
        _update_recording_timer()
    else:
        pauseButton.configure(text=i18n[current_ui_language]["pause_recording"])
        _update_recording_timer()

def stop_recording(save: bool = True):
    global recording_stream, is_recording, is_paused, audio_file_path
    if not is_recording:
        return
    is_recording = False
    is_paused = False
    if recording_stream:
        recording_stream.stop()
        recording_stream.close()
        recording_stream = None
    recordButton.configure(text=i18n[current_ui_language]["record"])
    pauseButton.configure(state="disabled")
    pauseButton.configure(text=i18n[current_ui_language]["pause_recording"])
    progress_bar.stop()
    progress_bar.pack_forget()
    progress_bar.configure(mode="determinate")
    progress_label.pack_forget()
    _stop_recording_timer()
    recording_start_time = None
    if save and recording_chunks:
        audio = np.concatenate(recording_chunks, axis=0)
        recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        filename = f"recording-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        filepath = os.path.join(recordings_dir, filename)
        sf.write(filepath, audio, 16000)
        audio_file_path = filepath
        config_data["last_audio_file"] = audio_file_path
        save_config()
        selected_file_label.configure(text=filename)
        result_textbox.delete("1.0", tkinter.END)
        result_textbox.insert(tkinter.END, i18n[current_ui_language]["recording_saved"].format(name=filename))

def transcribe():
    if not audio_file_path:
        result_textbox.delete("1.0", tkinter.END)
        result_textbox.insert(tkinter.END, i18n[current_ui_language]["select_file_prompt"])
        return
    
    engine_choice = engineOptionMenu.get()
    model_choice = modelOptionMenu.get()
    if config_data.get("use_external_server"):
        transcribe_thread = threading.Thread(target=_do_transcribe_external, args=(engine_choice, model_choice), daemon=True)
        transcribe_thread.start()
        return
    if engine_choice == "Whisper":
        if not is_whisper_model_downloaded(model_choice):
            size_text = get_whisper_model_size_text(model_choice)
            if not prompt_model_download(model_choice, size_text):
                result_textbox.delete("1.0", tkinter.END)
                result_textbox.insert(tkinter.END, i18n[current_ui_language]["download_cancelled"])
                return
    else:
        if not is_vosk_model_downloaded(model_choice):
            size_text = get_vosk_model_info(model_choice).get("size", i18n[current_ui_language]["unknown_size"])
            if not prompt_model_download(model_choice, size_text):
                result_textbox.delete("1.0", tkinter.END)
                result_textbox.insert(tkinter.END, i18n[current_ui_language]["download_cancelled"])
                return
    
    transcribe_thread = threading.Thread(target=_do_transcribe, args=(engine_choice, model_choice))
    transcribe_thread.start()

def send_transcription_request(engine_choice: str, model_choice: str, token: str) -> str:
    global remote_transcribe_job_id
    url = build_server_url("/v1/transcribe")
    boundary = "----voicepectaBoundary" + os.urandom(8).hex()
    body = io.BytesIO()

    model_to_send = model_choice
    if engine_choice.lower() == "vosk":
        model_info = get_vosk_model_info(model_choice)
        model_to_send = model_info.get("folder", model_choice)

    fields = {
        "engine": engine_choice,
        "model": model_to_send,
        "token": token,
    }
    language_code = get_language_code()
    if language_code:
        fields["language"] = language_code

    for name, value in fields.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))

    with open(audio_file_path, "rb") as f:
        file_content = f.read()

    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(audio_file_path)}"\r\n'.encode("utf-8")
    )
    body.write(b"Content-Type: application/octet-stream\r\n\r\n")
    body.write(file_content)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))

    data = body.getvalue()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Authorization", f"Bearer {token}")

    result_text = ""
    current_event = None
    data_lines = []

    with urllib.request.urlopen(req) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                if current_event == "result":
                    result_text = "\n".join(data_lines)
                elif current_event == "error":
                    raise RuntimeError("\n".join(data_lines) or "server error")
                current_event = None
                data_lines = []
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                if current_event == "status":
                    progress_label.configure(text=payload)
                    root.update_idletasks()
                if current_event == "job_id":
                    remote_transcribe_job_id = payload
                data_lines.append(payload)

    if result_text:
        return result_text
    raise RuntimeError("No result received from server")


def _do_transcribe_external(engine_choice: str, model_choice: str):
    global remote_transcribe_job_id, remote_transcribe_active
    lang_dict = i18n[current_ui_language]
    transcribeButton.configure(text=lang_dict["transcribing"], state="disabled")
    result_textbox.delete("1.0", tkinter.END)
    
    progress_bar.pack(pady=(5, 10), padx=10, fill="x")
    progress_label.pack(pady=(0, 10), padx=10)
    progress_bar.configure(mode="determinate")
    progress_bar.set(0)

    token = config_data.get("auth_token")
    if not token:
        result_textbox.insert(tkinter.END, i18n[current_ui_language]["login_first"])
        transcribeButton.configure(text=lang_dict["transcribe"], state="normal")
        progress_bar.pack_forget()
        progress_label.pack_forget()
        return

    remote_transcribe_active = True
    remote_transcribe_job_id = None
    try:
        result_text = send_transcription_request(engine_choice, model_choice, token)
        result_textbox.insert(tkinter.END, result_text)
    except Exception as e:
        log.error(f"External transcription failed: {e}")
        result_textbox.insert(tkinter.END, lang_dict["error_transcription"].format(e=e))
    finally:
        remote_transcribe_active = False
        remote_transcribe_job_id = None
        transcribeButton.configure(text=lang_dict["transcribe"], state="normal")
        progress_bar.pack_forget()
        progress_label.pack_forget()

def _do_transcribe(engine_choice: str, model_choice: str):
    lang_dict = i18n[current_ui_language]
    transcribeButton.configure(text=lang_dict["transcribing"], state="disabled")
    result_textbox.delete("1.0", tkinter.END)
    
    progress_bar.pack(pady=(5, 10), padx=10, fill="x")
    progress_label.pack(pady=(0, 10), padx=10)
    progress_bar.configure(mode="determinate")
    progress_bar.set(0)
    
    try:
        if engine_choice == "Whisper":
            whisper = get_whisper()
            torch = get_torch()
            original_stderr = sys.stderr
            sys.stderr = ProgressIOWrapper(original_stderr, progress_bar, progress_label, mode="download")
            model = None
            try:
                progress_label.configure(text=lang_dict["downloading_model"])
                device = "cpu" if cpu_checkbox.get() else ("cuda" if torch.cuda.is_available() else "cpu")
                model = whisper.load_model(model_choice, device=device)
            finally:
                sys.stderr = original_stderr

            original_stdout = sys.stdout
            try:
                progress_label.configure(text=lang_dict["loading_audio"])
                progress_bar.set(0)
                root.update_idletasks()
                
                audio = whisper.load_audio(audio_file_path)
                duration = len(audio) / whisper.audio.SAMPLE_RATE
                
                sys.stdout = ProgressIOWrapper(original_stdout, progress_bar, progress_label, mode="transcribe", total_duration=duration)
                
                language_code = get_language_code()
                result = model.transcribe(audio, language=language_code, verbose=True)
                result_textbox.insert(tkinter.END, format_whisper_result_text(result))
            finally:
                sys.stdout = original_stdout
        else:
            whisper = get_whisper()
            vosk = get_vosk()
            if not is_vosk_model_downloaded(model_choice):
                download_and_extract_vosk(model_choice)
            progress_label.configure(text=lang_dict["loading_audio"])
            progress_bar.set(0)
            root.update_idletasks()
            audio = whisper.load_audio(audio_file_path)
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            total_samples = len(audio_int16)
            model_info = get_vosk_model_info(model_choice)
            model_dir = os.path.join(VOSK_MODEL_DIR, model_info["folder"])
            vosk.SetLogLevel(-1)
            recognizer = vosk.KaldiRecognizer(vosk.Model(model_dir), whisper.audio.SAMPLE_RATE)
            chunk_size = 4000
            for i in range(0, total_samples, chunk_size):
                chunk = audio_int16[i:i + chunk_size]
                recognizer.AcceptWaveform(chunk.tobytes())
                processed_samples = min(i + chunk_size, total_samples)
                progress = processed_samples / total_samples if total_samples else 1.0
                progress_bar.set(progress)
                prefix = i18n[current_ui_language].get("transcribing_prefix", "Transcribing: ")
                percent_str = f"{int(progress * 100)}%"
                progress_label.configure(text=f"{prefix}{percent_str}")
                root.update_idletasks()
            final_result = recognizer.FinalResult()
            try:
                result_json = json.loads(final_result)
                result_textbox.insert(tkinter.END, result_json.get("text", ""))
            except json.JSONDecodeError:
                result_textbox.insert(tkinter.END, final_result)
    except Exception as e:
        log.error(f"Error during transcription: {e}")
        result_textbox.insert(tkinter.END, lang_dict["error_transcription"].format(e=e))
    finally:
        transcribeButton.configure(text=lang_dict["transcribe"], state="normal")
        progress_bar.pack_forget()
        progress_label.pack_forget()

# --- UI Setup ---
initial_theme = config_data.get("theme", "Dark")
try:
    customtkinter.set_appearance_mode(initial_theme)
except Exception as e:
    log.warning(f"Could not set initial appearance mode {initial_theme}: {e}")
    customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.geometry("760x635")
root.minsize(435, 620)
root.title(i18n[current_ui_language]["title"])
root.resizable(True, True)


def handle_app_close():
    if remote_transcribe_active:
        if not messagebox.askyesno(i18n[current_ui_language]["title"], i18n[current_ui_language]["quit_prompt_transcribing"]):
            return
        if remote_transcribe_job_id:
            cancel_remote_job(remote_transcribe_job_id)
    root.destroy()


root.protocol("WM_DELETE_WINDOW", handle_app_close)

# Load custom theme
# try:
    # root.tk.call("source", "assets/theme/dark.tcl")
    # apply_ttk_theme(initial_theme)
# except Exception as e:
#     logging.warning(f"Could not load custom theme: {e}")

# --- Main layout frames ---
left_frame = customtkinter.CTkFrame(root, width=250)
left_frame.pack(side="left", fill="y", padx=(10, 5), pady=10)
left_frame.pack_propagate(False)

right_frame = customtkinter.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

# --- Left frame widgets ---
img_for_ratio = Image.open("assets/logo.png")
original_width, original_height = img_for_ratio.size
aspect_ratio = original_height / original_width
new_width = 200
new_height = int(new_width * aspect_ratio)

logo_image = customtkinter.CTkImage(light_image=Image.open("assets/logo-dark.png"),
                                  dark_image=Image.open("assets/logo.png"),
                                  size=(new_width, new_height))
logo_label = customtkinter.CTkLabel(left_frame, image=logo_image, text="")
logo_label.pack(pady=20, padx=10)

selectFileButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["select_audio_file"], command=select_audio_file)
selectFileButton.pack(pady=(0, 10), padx=10, fill="x")

selected_file_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["no_file_selected"], wraplength=230, justify="center")
if audio_file_path:
    selected_file_label.configure(text=os.path.basename(audio_file_path))
selected_file_label.pack(pady=5, padx=10)

engine_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["engine"])
engine_label.pack(pady=(10, 0), padx=10)

initial_engine = config_data.get("engine", "Whisper")
if initial_engine not in ENGINE_OPTIONS:
    initial_engine = "Whisper"

engineOptionMenu = customtkinter.CTkComboBox(left_frame, values=ENGINE_OPTIONS, command=update_model_options)
engineOptionMenu.set(initial_engine)
engineOptionMenu.pack(pady=(0, 5), padx=10, fill="x")

lang_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["language_label"])
lang_label.pack(pady=(10, 0), padx=10)
LANG_OPTIONS = ["Russian", "Auto", "English"]
initial_lang = config_data.get("transcription_language", "Russian")
if initial_lang not in LANG_OPTIONS:
    initial_lang = "Russian"

def on_language_selected(choice: str):
    config_data["transcription_language"] = choice
    save_config()

language_combo = customtkinter.CTkComboBox(left_frame, values=LANG_OPTIONS, command=on_language_selected)
language_combo.set(initial_lang)
language_combo.pack(pady=(0, 5), padx=10, fill="x")

model_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["whisper_model"])
model_label.pack(pady=(10, 0), padx=10)

if initial_engine == "Vosk":
    model_values = list(VOSK_MODELS.keys())
    initial_model = config_data.get("model", "base model")
    if initial_model not in model_values:
        initial_model = model_values[0]
else:
    model_values = WHISPER_MODELS
    initial_model = config_data.get("model", "base")
    if initial_model not in model_values:
        initial_model = "base"

modelOptionMenu = customtkinter.CTkComboBox(left_frame, values=model_values, command=on_model_selected)
modelOptionMenu.set(initial_model)
modelOptionMenu.pack(pady=(0, 5), padx=10, fill="x")

cpu_checkbox = customtkinter.CTkCheckBox(left_frame, text=i18n[current_ui_language]["use_cpu"], command=on_cpu_toggle)
if config_data.get("use_cpu"):
    cpu_checkbox.select()
cpu_checkbox.pack(pady=10, padx=10)

if initial_engine == "Vosk":
    model_label.configure(text=i18n[current_ui_language]["vosk_model"])
    cpu_checkbox.configure(state="disabled")

transcribeButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["transcribe"], command=transcribe, height=48)
transcribeButton.pack(side="bottom", pady=10, padx=10, fill="x")

pauseButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["pause_recording"], command=toggle_pause, state="disabled")
pauseButton.pack(side="bottom", pady=(0, 10), padx=10, fill="x")

recordButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["record"], command=toggle_recording)
recordButton.pack(side="bottom", pady=(0, 10), padx=10, fill="x")

settingsButton = customtkinter.CTkButton(
    left_frame,
    text=i18n[current_ui_language]["settings"],
    command=open_settings_window,
    fg_color="transparent",
    border_width=2,
    text_color=("black", "white"),
)
settingsButton.pack(side="bottom", pady=(0,10), padx=10, fill="x")

# --- Right frame widgets ---
result_textbox = customtkinter.CTkTextbox(right_frame, wrap="word")
account_button = customtkinter.CTkButton(right_frame, text=i18n[current_ui_language]["account"], command=show_account_dialog)
account_button.pack(fill="x", padx=5, pady=(5, 0))

result_textbox = customtkinter.CTkTextbox(right_frame, wrap="word")
result_textbox.pack(fill="both", expand=True, padx=5, pady=5)

progress_bar = customtkinter.CTkProgressBar(right_frame, mode="determinate")
progress_label = customtkinter.CTkLabel(right_frame, text="")


def show_initial_dialog():
    global startup_dialog_active
    if startup_dialog_active:
        return
    startup_dialog_active = True

    dialog = customtkinter.CTkToplevel(root)
    dialog.title(i18n[current_ui_language]["title"])
    dialog.geometry("480x380")
    dialog.transient(root)
    dialog.lift()
    dialog.update_idletasks()
    dialog.wait_visibility()
    dialog.grab_set()
    dialog.focus_force()
    dialog.bind("<Escape>", lambda _e: on_close())

    def on_close():
        nonlocal dialog
        if messagebox.askyesno(i18n[current_ui_language]["title"], i18n[current_ui_language]["quit_prompt"]):
            root.destroy()
        else:
            dialog.deiconify()
    dialog.protocol("WM_DELETE_WINDOW", on_close)

    customtkinter.CTkLabel(dialog, text=i18n[current_ui_language]["login_dialog_title"], font=("Arial", 20)).pack(pady=(25, 15))

    form_frame = customtkinter.CTkFrame(dialog, fg_color="transparent")
    form_frame.pack(pady=10, padx=25, fill="x")

    login_row = customtkinter.CTkFrame(form_frame, fg_color="transparent")
    login_row.pack(fill="x", pady=(0, 10))
    customtkinter.CTkLabel(login_row, text=i18n[current_ui_language]["login_label"], font=("Arial", 16), width=90, anchor="e").grid(row=0, column=0, padx=(0, 8))
    login_entry = customtkinter.CTkEntry(login_row, width=260)
    login_entry.grid(row=0, column=1, sticky="ew")
    login_row.grid_columnconfigure(1, weight=1)

    password_row = customtkinter.CTkFrame(form_frame, fg_color="transparent")
    password_row.pack(fill="x", pady=(0, 10))
    customtkinter.CTkLabel(password_row, text=i18n[current_ui_language]["password_label"], font=("Arial", 16), width=90, anchor="e").grid(row=0, column=0, padx=(0, 8))
    password_entry = customtkinter.CTkEntry(password_row, show="*", width=260)
    password_entry.grid(row=0, column=1, sticky="ew")
    password_row.grid_columnconfigure(1, weight=1)

    status_label = customtkinter.CTkLabel(dialog, text="", text_color="#bbbbbb")
    status_label.pack()

    def handle_login():
        global startup_dialog_active
        login = login_entry.get().strip()
        password = password_entry.get().strip()
        if not login or not password:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["enter_login_password"])
            return
        status_label.configure(text=i18n[current_ui_language]["contacting_server"])
        dialog.update_idletasks()
        ok, reason = attempt_login_remote(login, password)
        if ok:
            status_label.configure(text=i18n[current_ui_language]["login_successful"])
            dialog.destroy()
            startup_dialog_active = False
        else:
            status_label.configure(text="")
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["login_failed"].format(reason=reason))

    customtkinter.CTkButton(dialog, text=i18n[current_ui_language]["login_button"], command=handle_login, width=320).pack(pady=(10, 12))

    register_link = customtkinter.CTkLabel(dialog, text=i18n[current_ui_language]["register_link"], text_color="#3b4acb",
                                           cursor="hand2", font=("Arial", 16, "underline"))
    register_link.pack(pady=(5, 8))
    def go_register(_e=None):
        global startup_dialog_active
        dialog.destroy()
        startup_dialog_active = False
        show_register_dialog()
    register_link.bind("<Button-1>", go_register)

    ttk.Separator(dialog, orient="horizontal").pack(fill="x", pady=(12, 12), padx=20)

    def handle_local():
        global startup_dialog_active
        nonlocal dialog
        if check_local_dependencies():
            config_data.update({
                "use_external_server": False,
                "auth_token": None,
                "onboarded": True,
            })
            save_config()
            dialog.destroy()
            startup_dialog_active = False
        else:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["models_not_installed"])
            dialog.destroy()
            startup_dialog_active = False
            root.after(50, show_initial_dialog)

    customtkinter.CTkButton(dialog, text=i18n[current_ui_language]["use_local_version"], command=handle_local, width=340).pack(pady=(12, 16))

def show_register_dialog():
    global startup_dialog_active
    reg = customtkinter.CTkToplevel(root)
    reg.title(i18n[current_ui_language]["register_dialog_title"])
    reg.geometry("460x340")
    reg.transient(root)
    reg.lift()
    reg.update_idletasks()
    reg.wait_visibility()
    reg.grab_set()
    reg.focus_force()

    def close_register(_e=None):
        global startup_dialog_active
        reg.destroy()
        startup_dialog_active = False
        root.after(50, show_initial_dialog)

    reg.bind("<Escape>", close_register)

    customtkinter.CTkLabel(reg, text=i18n[current_ui_language]["register_dialog_title"], font=("Arial", 20)).pack(pady=(20, 12))

    form_frame = customtkinter.CTkFrame(reg, fg_color="transparent")
    form_frame.pack(pady=10, padx=25, fill="x")

    login_row = customtkinter.CTkFrame(form_frame, fg_color="transparent")
    login_row.pack(fill="x", pady=(0, 10))
    customtkinter.CTkLabel(login_row, text=i18n[current_ui_language]["login_label"], font=("Arial", 16), width=90, anchor="e").grid(row=0, column=0, padx=(0, 8))
    login_entry = customtkinter.CTkEntry(login_row, width=240)
    login_entry.grid(row=0, column=1, sticky="ew")
    login_row.grid_columnconfigure(1, weight=1)

    password_row = customtkinter.CTkFrame(form_frame, fg_color="transparent")
    password_row.pack(fill="x", pady=(0, 12))
    customtkinter.CTkLabel(password_row, text=i18n[current_ui_language]["password_label"], font=("Arial", 16), width=90, anchor="e").grid(row=0, column=0, padx=(0, 8))
    password_entry = customtkinter.CTkEntry(password_row, show="*", width=240)
    password_entry.grid(row=0, column=1, sticky="ew")
    password_row.grid_columnconfigure(1, weight=1)

    status_label = customtkinter.CTkLabel(reg, text="", text_color="#bbbbbb")
    status_label.pack()

    def submit_registration():
        global startup_dialog_active
        login = login_entry.get().strip()
        password = password_entry.get().strip()
        if not login or not password:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["enter_login_password"])
            return
        status_label.configure(text=i18n[current_ui_language]["contacting_server"])
        reg.update_idletasks()
        ok, reason = attempt_register_remote(login, password)
        if ok:
            messagebox.showinfo(i18n[current_ui_language]["title"], i18n[current_ui_language]["registration_success"])
            startup_dialog_active = False
            reg.destroy()
            show_initial_dialog()
        else:
            status_label.configure(text="")
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["registration_failed"].format(reason=reason))

    customtkinter.CTkButton(reg, text=i18n[current_ui_language]["register_button"], command=submit_registration, width=300).pack(pady=(6, 10))

    login_link = customtkinter.CTkLabel(reg, text=i18n[current_ui_language]["login_link"], text_color="#3b4acb", cursor="hand2",
                                        font=("Arial", 16, "underline"))
    login_link.pack(pady=(4, 8))
    login_link.bind("<Button-1>", lambda _e: (reg.destroy(), show_initial_dialog()))

    ttk.Separator(reg, orient="horizontal").pack(fill="x", pady=(8, 10), padx=20)

    def handle_local_reg():
        if check_local_dependencies():
            config_data.update({
                "use_external_server": False,
                "auth_token": None,
                "onboarded": True,
            })
            save_config()
            reg.destroy()
            startup_dialog_active = False
        else:
            messagebox.showerror(i18n[current_ui_language]["title"], i18n[current_ui_language]["models_not_installed"])
            reg.destroy()
            startup_dialog_active = False
            root.after(50, show_initial_dialog)

    customtkinter.CTkButton(reg, text=i18n[current_ui_language]["use_local_version"], command=handle_local_reg, width=300).pack(pady=(6, 14))


def maybe_show_startup_dialog():
    need_dialog = not config_data.get("onboarded", False)
    if config_data.get("use_external_server"):
        token = config_data.get("auth_token")
        if not (token and validate_token(token)):
            need_dialog = True
    if need_dialog:
        root.after(150, show_initial_dialog)

# --- Logging ---
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

maybe_show_startup_dialog()
log.info("App Ready")
root.mainloop()
