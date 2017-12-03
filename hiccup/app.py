from tkinter import Button, OptionMenu, StringVar
from tkinter.filedialog import askopenfilename, askdirectory

import hiccup.model as model
import hiccup.run as run

"""
Any crap UI stuff I build for belch
"""


class BelchUI:
    def __init__(self, master):
        self.master = master
        master.title("Belch")

        self.input_img_button = Button(master, text="Choose Input Image", command=self.get_img)
        self.input_img_button.pack()

        self.output_dir_button = Button(master, text="Choose the output directory", command=self.get_dir)
        self.output_dir_button.pack()

        self.style_var = StringVar(master)
        self.style_dd = OptionMenu(master, self.style_var, *{
            model.Compression.JPEG.value, model.Compression.HIC.value
        })
        self.style_dd.pack()

        self.compress_button = Button(master, text="Compress", command=self.compress)
        self.compress_button.pack()

        self.decompress_button = Button(master, text="Decompress", command=self.decompress)
        self.decompress_button.pack()

        self.filename = None
        self.dir = None

    def get_dir(self):
        self.dir = askdirectory()

    def get_img(self):
        self.filename = askopenfilename()

    def decompress(self):
        run.decompress(self.filename)

    def compress(self):
        run.compress(self.filename, self.dir, model.Compression(self.style_var.get()))
