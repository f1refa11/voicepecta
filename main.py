import tkinter
import tkinter.ttk
from tkinter import filedialog
import customtkinter
from PIL import Image
import whisper
import logging
import threading
import sys
import io
import re
import torch

# --- Globals ---
audio_file_path = None
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
    }
}

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
    model_label.configure(text=lang_dict["whisper_model"])
    transcribeButton.configure(text=lang_dict["transcribe"])
    settingsButton.configure(text=lang_dict["settings"])
    cpu_checkbox.configure(text=lang_dict["use_cpu"])

def change_theme(new_theme: str):
    customtkinter.set_appearance_mode(new_theme)

def open_settings_window():
    settings_win = customtkinter.CTkToplevel(root)
    settings_win.title(i18n[current_ui_language]["settings_title"])
    settings_win.geometry("350x200")
    settings_win.transient(root)

    def update_settings_ui(lang_choice: str):
        lang_code = "ru" if lang_choice == "Russian" else "en"
        update_ui_language(lang_choice)
        settings_win.title(i18n[lang_code]["settings_title"])
        lang_label.configure(text=i18n[lang_code]["ui_language"])
        theme_label.configure(text=i18n[lang_code]["theme"])

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
    
    settings_win.grab_set()

def select_audio_file():
    global audio_file_path
    audio_file_path = filedialog.askopenfilename()
    if audio_file_path:
        selected_file_label.configure(text=audio_file_path.split("/")[-1])

def transcribe():
    if not audio_file_path:
        result_textbox.delete("1.0", tkinter.END)
        result_textbox.insert(tkinter.END, i18n[current_ui_language]["select_file_prompt"])
        return
    
    transcribe_thread = threading.Thread(target=_do_transcribe)
    transcribe_thread.start()

def _do_transcribe():
    lang_dict = i18n[current_ui_language]
    transcribeButton.configure(text=lang_dict["transcribing"], state="disabled")
    result_textbox.delete("1.0", tkinter.END)
    
    progress_bar.pack(pady=(5, 10), padx=10, fill="x")
    progress_label.pack(pady=(0, 10), padx=10)
    progress_bar.set(0)
    
    original_stderr = sys.stderr
    sys.stderr = ProgressIOWrapper(original_stderr, progress_bar, progress_label, mode="download")
    
    model = None
    try:
        model_name = modelOptionMenu.get()
        progress_label.configure(text=lang_dict["downloading_model"])
        
        device = "cpu" if cpu_checkbox.get() else ("cuda" if torch.cuda.is_available() else "cpu")
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        log.error(f"Error loading model: {e}")
        result_textbox.insert(tkinter.END, f"Error loading model: {e}")
        sys.stderr = original_stderr
        transcribeButton.configure(text=lang_dict["transcribe"], state="normal")
        progress_bar.pack_forget()
        progress_label.pack_forget()
        return
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

    except Exception as e:
        log.error(f"Error during transcription: {e}")
        result_textbox.insert(tkinter.END, lang_dict["error_transcription"].format(e=e))
    finally:
        sys.stdout = original_stdout
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

model_label = customtkinter.CTkLabel(left_frame, text=i18n[current_ui_language]["whisper_model"])
model_label.pack(pady=(20, 5), padx=10)

modelOptionMenu = customtkinter.CTkComboBox(left_frame, values=["tiny", "base", "small", "medium", "large"])
modelOptionMenu.set("base")
modelOptionMenu.pack(pady=5, padx=10, fill="x")

cpu_checkbox = customtkinter.CTkCheckBox(left_frame, text=i18n[current_ui_language]["use_cpu"])
cpu_checkbox.pack(pady=10, padx=10)

transcribeButton = customtkinter.CTkButton(left_frame, text=i18n[current_ui_language]["transcribe"], command=transcribe)
transcribeButton.pack(side="bottom", pady=10, padx=10, fill="x")

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