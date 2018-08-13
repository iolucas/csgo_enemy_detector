from win32api import GetKeyState
import win32api
import time


special_keys = [0x01, 0x02, 0x10, 0x20]

special = {0x01: 'leftClick',
           0x02: 'rightClick',
           0x10: 'shift',
           0x20: 'space'}

time.sleep(1)
while True:
    for i in range(1, 256):
        if win32api.GetAsyncKeyState(i):
            if i in special_keys:
                print(special[i])
            else:
                print(chr(i))
    time.sleep(0.01)