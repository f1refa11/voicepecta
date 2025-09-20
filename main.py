import tkinter
import tkinter.ttk

import PIL
from PIL import Image, ImageTk
import customtkinter
import whisper
import logging

def transcribe():
	log.info("Transcribe function invoked")
	
	model = whisper.load_model("turbo")
	log.info("Model loaded")
	
	result = model.transcribe("audio.mp3")
	log.info("Transcribe completed")
	return result

def resize_canvas(event):
	new_width = event.width
	new_height = event.height
	
	# Update the canvas size
	logoCanvas.config(width = 690, height = 144)
	# logoCanvas.itemconfigure(, image = photob)  # Update the image size (optional)

VERSION="0.0.0"

root = customtkinter.CTk()
root.geometry("500x500")
root.title(f"voicepecta v{VERSION}")
root.resizable(True, True)

style = tkinter.ttk.Style(master=root)
style.tk.call("source", "assets/theme/sv.tcl")
style.theme_use("sun-valley-dark")

logoCanvas = tkinter.Canvas(root, width=348, height=442)
logoImage = tkinter.PhotoImage(file="assets/logo.png")
logoCanvas.create_image(0, 0, anchor="nw", image=logoImage)
# logoImageLabel = customtkinter.CTkLabel(root, image=logoImage, text="")
# logoImageLabel.grid(column=0, row=0, padx=10, pady=10, sticky=tkinter.E)
logoCanvas.grid(column=0, row=0, padx=10, pady=10, sticky=tkinter.E)



modelOptionMenu = tkinter.ttk.Combobox(root, values=["tiny", "base", "small", "medium", "large", "turbo"])
modelOptionMenu.set("turbo")
modelOptionMenu.grid(column=0, row=1, padx=10, pady=10, sticky=tkinter.E)

languageOptionMenu = tkinter.ttk.Combobox(root, values=["Russian", "English"])
languageOptionMenu.set("Russian")
languageOptionMenu.grid(column=0, row=2, padx=10, pady=10, sticky=tkinter.E)

transcribeButton = tkinter.ttk.Button(root, text="Transcribe", command=transcribe)
transcribeButton.grid(column=0, row=3, padx=10, pady=10, sticky=tkinter.E)

root.bind('<Configure>', resize_canvas)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.info("App Ready")
# log.addHandler(logging.FileHandler())
root.mainloop()