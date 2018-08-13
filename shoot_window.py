import tkinter as tk
import numpy as np
#from tkinter import *
from PIL import Image, ImageGrab, ImageTk
from io import BytesIO

#https://stackoverflow.com/questions/40019449/python-tkinter-displaying-images-as-movie-stream

def np_to_tkimage(np_image):
    pil_image = Image.frombytes('RGB', (np_image.shape[1], np_image.shape[0]), np_image.astype('b').tostring())
    tk_image = ImageTk.PhotoImage(image = pil_image)
    return tk_image


class ShootWindow:
    def __init__(self, loop_func, delay):
        root = tk.Tk()
        root.wm_attributes("-topmost", 1) #Keep always on top

        #Create label
        label_text = tk.StringVar()
        label = tk.Label(root, textvariable=label_text)
        label.pack()
        label_text.set("Initializing...")

        #Create frame
        frame = tk.Frame(root, width=302, height=302)
        frame.pack()

        #Create canvas
        canvas = tk.Canvas(frame, width=300,height=300)
        canvas.place(x=-0,y=-0)

        img = np.ones((300,300,3))*200
        tk_img = np_to_tkimage(img)

        canvas_image = canvas.create_image(0, 0, image = tk_img, anchor = tk.NW)

        self.root = root
        self.canvas = canvas
        self.canvas_image = canvas_image

        self.label_text = label_text

        root.after("idle", lambda: self._repeat(loop_func, delay))

        root.mainloop()

    def _repeat(self, loop_func, delay):
        loop_func(self)
        self.root.after(delay, lambda: self._repeat(loop_func, delay))

    def set_image(self, image):
        self.img_buffer = np_to_tkimage(image)
        self.canvas.itemconfig(self.canvas_image, image=self.img_buffer)
        
    def set_text(self, text):
        self.label_text.set(text)



if __name__ == "__main__":

    def loop_func(sw_self):
        img = np.ones((300,300,3))*np.random.randint(0,255)

        sw_self.set_image(img)
        sw_self.set_text(np.random.random())

    sw = ShootWindow(loop_func, 100)



class buildFrame:
    def __init__(self):
        #Create root element
        self.root = tk.Tk()
        



        #Create frame
        #frame1 = tk.Frame(self.root, width=302, height=302)
        #frame1.pack()

        #Create canvas
        #canvas1 = tk.Canvas(self.root, width=300, height=300)
        #canvas1.pack()
        
        #self.canvas1 = canvas1
        
        #tmp = Image.new('RGB', (300, 300), color=(0, 0, 0))
        #tmp = numpy2image(np.ones((300,300,3))*200)
        #imgObj = ImageTk.PhotoImage(image=tmp)
        #self.thsObj = canvas1.create_image(0,0, anchor=tk.NW, image=imgObj)

        frame = tk.Frame(self.root, width=302, height=302)
        frame.pack()

        canvas = tk.Canvas(frame, width=300,height=300)
        canvas.place(x=-0,y=-0)

        #data = np.array(np.random.random((400,500))*100,dtype=int)

        img = np.ones((300,300,3))*128

        im = numpy2image(img)


        photo = ImageTk.PhotoImage(image = im)

        canvas.create_image(0, 0, image = photo, anchor = tk.NW)

        self.root.update()
        
        #canvas1.itemconfig(thsObj, image=imgObj)
        self.root.mainloop()
        #self.root.after("idle", self.snapS)

    def snapS(self):
        quality_val = 70
        mem_file = BytesIO()
        ImageGrab.grab().save(mem_file, format="JPEG", quality=quality_val)
        mem_file.seek(0)
        tmp = Image.open(mem_file)
        tmp.thumbnail([800, 600])

        img = np.ones((300,300,3))*128

        im = numpy2image(img)
        photo = ImageTk.PhotoImage(image = im)


        self.image = ImageTk.PhotoImage(tmp)
        self.canvas1.itemconfig(self.thsObj, image=photo)
        mem_file.close()
        self.root.after(10, self.snapS)

#world = buildFrame()
#world.root.mainloop()



