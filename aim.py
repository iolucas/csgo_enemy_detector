import tkinter as tk
import numpy as np
#from tkinter import *
from PIL import Image, ImageGrab, ImageTk
#from io import BytesIO


def np_to_tkimage(np_image):
    pil_image = Image.frombytes('RGB', (np_image.shape[1], np_image.shape[0]), np_image.astype('b').tostring())
    tk_image = ImageTk.PhotoImage(image = pil_image)
    return tk_image



root = tk.Tk()

import win32api

screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)

aim_width, aim_height = 316,316
#canvas.place(x=-0,y=-0)

aim_img = np.zeros((aim_width, aim_height,3))
aim_img[:,:,1] = np.ones((aim_width, aim_height))*255

aim_img_empty = np.ones((302,302,3))*255

aim_img[7:-7, 7:-7, :] = aim_img_empty


root.image = np_to_tkimage(aim_img)

# The image must be stored to Tk or it will be garbage collected.
label = tk.Label(root, image=root.image, bg='white')
root.overrideredirect(True)
#root.geometry("+0+0")
root.geometry("+{}+{}".format(int((screen_width-aim_width)/2), int((screen_height-aim_height)/2)))
root.lift()
root.wm_attributes("-topmost", True)
#root.wm_attributes("-disabled", True)
root.wm_attributes("-transparentcolor", "white")
label.pack()
label.mainloop()