import numpy as np

import tkinter as tk
from tkinter import *

from PIL import Image, ImageTk

def numpy2image(np_img):
    return Image.frombytes('RGB', (np_img.shape[1], np_img.shape[0]), np_img.astype('b').tostring())


img = np.ones((300,300,3))*128

im = numpy2image(img)




root = Tk()
root.wm_attributes("-topmost", 1)

v = StringVar()
w = tk.Label(root, textvariable=v)
w.pack()


v.set("lucas")


frame = tk.Frame(root, width=302, height=302)
frame.pack()

canvas = tk.Canvas(frame, width=300,height=300)
canvas.place(x=-0,y=-0)

#data = np.array(np.random.random((400,500))*100,dtype=int)



photo = ImageTk.PhotoImage(image = im)

canvas.create_image(0, 0, image = photo, anchor = tk.NW)

root.update()

root.mainloop()