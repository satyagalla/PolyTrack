import pyautogui
import pydirectinput
import time

pyautogui.FAILSAFE = True

time.sleep(3)

while True:
    pydirectinput.keyDown('w')