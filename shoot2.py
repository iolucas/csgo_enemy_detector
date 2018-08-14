from model import ShootModel
from shoot_window import ShootWindow
from aim_window import AimWindow

from grabscreen import grab_screen
import cv2

import numpy as np

#import pyautogui

##import win32com.client as comctl
#wsh = comctl.Dispatch("WScript.Shell")

if __name__ == "__main__":

    red_mask = np.ones((300,300), dtype=np.uint8)*150

    #Remove outer frame and keep background translucid with red color if should shoot

    def aim_loop_func(sw_self):
        
        screen = grab_screen(region=(650,300,949,599))#left, top, x2, y2, subtract 1 from x2 and y2 to match 300x300pixels
        #cv2.imwrite("tmp123.jpg", screen)
        #screen = cv2.imread("tmp123.jpg")
        
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

        pred = sm.predict([screen])[0]
        #pred = 0

        sw_self.set_pred_value(round(pred,4))

        if pred > 0.7:
            sw_self.set_aim("red")
            sw_self.set_target_value("Enemy")
        else:
            sw_self.set_aim("green")
            sw_self.set_target_value("Nothing")


    def shoot_loop_func(sw_self):
        
        screen = grab_screen(region=(650,300,949,599))#left, top, x2, y2, subtract 1 from x2 and y2 to match 300x300pixels
        cv2.imwrite("tmp123.jpg", screen)
        screen = cv2.imread("tmp123.jpg")
        
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        pred = sm.predict([screen])[0]
        #pred = 0

        if pred > 0.5:
            pred_text = "FIRE"
            screen[:, :, 0] += red_mask
            #mx, my = pyautogui.position()
            #pyautogui.click(x=mx, y=my, clicks=1, interval=1, button='left') #Click mouse
        else:
            pred_text = 'halt'
            #wsh.SendKeys("l")
                    
        sw_self.set_image(screen)
        sw_self.set_text(pred_text)

    sm = ShootModel()

    #sw = ShootWindow(shoot_loop_func, 100)
    sw = AimWindow(aim_loop_func, 100)

    












