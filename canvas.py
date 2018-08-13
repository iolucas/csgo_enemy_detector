from tkinter import *
from PIL import Image, ImageGrab, ImageTk
from io import BytesIO

class buildFrame:
    def __init__(self):
        self.root = Tk()
        self.land = Canvas(self.root, width=800, height=600)
        self.land.pack()
        tmp = Image.new('RGBA', (800, 600), color=(0, 0, 0))
        self.imgObj = ImageTk.PhotoImage(image=tmp)
        self.thsObj = self.land.create_image(0,0, anchor='nw', image=self.imgObj)
        self.root.after("idle", self.snapS)

    def snapS(self):
        quality_val = 70
        mem_file = BytesIO()
        ImageGrab.grab().save(mem_file, format="JPEG", quality=quality_val)
        mem_file.seek(0)
        tmp = Image.open(mem_file)
        tmp.thumbnail([800, 600])
        self.image = ImageTk.PhotoImage(tmp)
        self.land.itemconfig(self.thsObj, image=self.image)
        mem_file.close()
        self.root.after(10, self.snapS)

world = buildFrame()
world.root.mainloop()