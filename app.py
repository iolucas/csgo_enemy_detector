from win32api import GetKeyState
import win32api
import time

import win32com.client as comctl
wsh = comctl.Dispatch("WScript.Shell")

from grabscreen import grab_screen
import cv2

import random

#0x01: 'leftClick',
#0x02: 'rightClick',
#0x10: 'shift',
#0x20: 'space'}

time.sleep(5)

last_state = False

cnt = 0

while True:
    cnt += 1

    delay = 0.01
    
    current_state = bool(win32api.GetAsyncKeyState(0x01))
    
    #If a click is detected
    #Check the past states of the mouse
    #if (last_state == True and current_state == False):
    if current_state:
        #wsh.SendKeys("{F12}")
        print("Clicked!")
        delay = 0.5
        
        screen = grab_screen(region=(650,300,949,599))#left, top, x2, y2, sub 1 from x2 and y2 to match 300x300pixels
        cv2.imwrite("imgs/pressed_" + str(time.time()) + ".jpg", screen)

    #roughly every 2 seconds, print screen
    elif cnt >= 100:
        #pass
        cnt = 0
        screen = grab_screen(region=(650,300,949,599))
        cv2.imwrite("imgs/not_pressed_" + str(time.time()) + ".jpg", screen)
    
    last_state = current_state

    time.sleep(delay)