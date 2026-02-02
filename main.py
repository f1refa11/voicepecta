import tkinter
import tkinter.ttk
from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image
import whisper
import logging
import threading
import sys
import io
import re
import torch
import os
import json
import urllib.request
import urllib.error
import zipfile
import numpy as np
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
from typing import Optional
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# --- Globals ---
audio_file_path = None
recording_stream = None
recording_chunks = []
is_recording = False
is_paused = False
selected_mic_index = None
VERSION = "0.0.4"
current_ui_language = "ru"

# --- Internationalization ---
i18n = {
    "en": {
        "title": f"voicepecta v{VERSION}",
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
    },
    "ru": {
        "title": f"voicepecta v{VERSION}",
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
    
    lang_dict = i18n[current_ui_language]
    root.title(lang_dict["title"])
    selectFileButton.configure(text=lang_dict["select_audio_file"])
    if not audio_file_path:
        selected_file_label.configure(text=lang_dict["no_file_selected"])
    update_model_label()
    engine_label.configure(text=lang_dict["engine"])
    recordButton.configure(text=lang_dict["record"])
    pauseButton.configure(text=lang_dict["pause_recording"] if not is_paused else lang_dict["resume_recording"])
    stopButton.configure(text=lang_dict["stop_recording"])
    transcribeButton.configure(text=lang_dict["transcribe"])
    settingsButton.configure(text=lang_dict["settings"])
    cpu_checkbox.configure(text=lang_dict["use_cpu"])

def change_theme(new_theme: str):
    customtkinter.set_appearance_mode(new_theme)

def open_settings_window():
    settings_win = customtkinter.CTkToplevel(root)
    settings_win.title(i18n[current_ui_language]["settings_title"])
    settings_win.geometry("350x260")
    settings_win.transient(root)

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
    return whisper._MODELS.get(model_name)

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
    if engine_choice == "Vosk":
        modelOptionMenu.configure(values=list(VOSK_MODELS.keys()))
        modelOptionMenu.set("base model")
        cpu_checkbox.configure(state="disabled")
    else:
        modelOptionMenu.configure(values=WHISPER_MODELS)
        modelOptionMenu.set("base")
        cpu_checkbox.configure(state="normal")
    update_model_label()

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

def _recording_callback(indata, frames, time_info, status):
    if status:
        log.warning(f"Recording status: {status}")
    if not is_recording or is_paused:
        return
    recording_chunks.append(indata.copy())

def start_recording():
    global recording_stream, recording_chunks, is_recording, is_paused, audio_file_path
    if is_recording:
        return
    recording_chunks = []
    is_recording = True
    is_paused = False
    recordButton.configure(state="disabled")
    pauseButton.configure(state="normal")
    stopButton.configure(state="normal")
    progress_label.configure(text=i18n[current_ui_language]["recording_status"])
    progress_label.pack(pady=(0, 10), padx=10)
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
        progress_label.configure(text=i18n[current_ui_language]["recording_paused"])
    else:
        pauseButton.configure(text=i18n[current_ui_language]["pause_recording"])
        progress_label.configure(text=i18n[current_ui_language]["recording_status"])

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
    recordButton.configure(state="normal")
    pauseButton.configure(state="disabled")
    pauseButton.configure(text=i18n[current_ui_language]["pause_recording"])
    stopButton.configure(state="disabled")
    progress_label.pack_forget()
    if save and recording_chunks:
        audio = np.concatenate(recording_chunks, axis=0)
        recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        filename = f"recording-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        filepath = os.path.join(recordings_dir, filename)
        sf.write(filepath, audio, 16000)
        audio_file_path = filepath
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

def _do_transcribe(engine_choice: str, model_choice: str):
    lang_dict = i18n[current_ui_language]
    transcribeButton.configure(text=lang_dict["transcribing"], state="disabled")
    result_textbox.delete("1.0", tkinter.END)
    
    progress_bar.pack(pady=(5, 10), padx=10, fill="x")
    progress_label.pack(pady=(0, 10), padx=10)
    progress_bar.set(0)
    
    try:
        if engine_choice == "Whisper":
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
                
                result = model.transcribe(audio, language="russian", verbose=True)
                result_textbox.insert(tkinter.END, result["text"])
            finally:
                sys.stdout = original_stdout
        else:
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
            SetLogLevel(-1)
            recognizer = KaldiRecognizer(VoskModel(model_dir), whisper.audio.SAMPLE_RATE)
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
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.geometry("750x500")
root.title(i18n[current_ui_language]["title"])
root.resizable(True, True)

# Load custom theme
try:
    root.tk.call("source", "assets/theme/dark.tcl")
    style = tkinter.ttk.Style()
    style.theme_use("sun-valley-dark")
except Exception as e:
    logging.warning(f"Could not load custom theme: {e}")

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
selectFileButton.pack(pady=10, padx=10, fill="x")

selected_file_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["no_file_selected"], wraplength=230, justify="center")
selected_file_label.pack(pady=5, padx=10)

engine_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["engine"])
engine_label.pack(pady=(20, 5), padx=10)

engineOptionMenu = customtkinter.CTkComboBox(left_frame, values=ENGINE_OPTIONS, command=update_model_options)
engineOptionMenu.set("Whisper")
engineOptionMenu.pack(pady=5, padx=10, fill="x")

model_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["whisper_model"])
model_label.pack(pady=(20, 5), padx=10)

modelOptionMenu = customtkinter.CTkComboBox(left_frame, values=WHISPER_MODELS)
modelOptionMenu.set("base")
modelOptionMenu.pack(pady=5, padx=10, fill="x")

cpu_checkbox = customtkinter.CTkCheckBox(left_frame, text=i18n[current_ui_language]["use_cpu"])
cpu_checkbox.pack(pady=10, padx=10)

transcribeButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["transcribe"], command=transcribe)
transcribeButton.pack(side="bottom", pady=10, padx=10, fill="x")

stopButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["stop_recording"], command=stop_recording, state="disabled")
stopButton.pack(side="bottom", pady=(0, 10), padx=10, fill="x")

pauseButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["pause_recording"], command=toggle_pause, state="disabled")
pauseButton.pack(side="bottom", pady=(0, 10), padx=10, fill="x")

recordButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["record"], command=start_recording)
recordButton.pack(side="bottom", pady=(0, 10), padx=10, fill="x")

settingsButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["settings"], command=open_settings_window, fg_color="transparent", border_width=2)
settingsButton.pack(side="bottom", pady=(0,10), padx=10, fill="x")

# --- Right frame widgets ---
result_textbox = customtkinter.CTkTextbox(right_frame, wrap="word")
result_textbox.pack(fill="both", expand=True, padx=5, pady=5)

progress_bar = customtkinter.CTkProgressBar(right_frame, mode="determinate")
progress_label = customtkinter.CTkLabel(right_frame, text="")

# --- Logging ---
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

log.info("App Ready")
root.mainloop()
