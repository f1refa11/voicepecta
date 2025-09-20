import customtkinter
import whisper

VERSION="0.0.0"

root = customtkinter.CTk()
root.geometry("500x500")
root.title(f"voicepecta v{VERSION}")
root.resizable(True, True)

root.mainloop()