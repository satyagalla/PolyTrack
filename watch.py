import mss
import cv2
import numpy as np
import pygetwindow as gw
import time

# 1. FIND THE WINDOW
# Change 'Polytrack' to whatever part of the browser title is unique (e.g., 'Google Chrome')
try:
    window = gw.getWindowsWithTitle('Edge')[0] # Get the first window that matches
    if window:
        # Activate it (bring to front) just to be safe
        # print(window)
        window.activate() 
except IndexError:
    print("Could not find the window! Is the game open?")
    exit()

with mss.mss() as sct:
    while True:
        # 2. UPDATE COORDINATES DYNAMICALLY
        # We do this inside the loop in case you move the window while playing
        monitor = {
            "top": window.top + 105, 
            "left": window.left + 15, 
            "width": window.width - 30, 
            "height": window.height - 120
        }

        # 3. CAPTURE
        img = np.array(sct.grab(monitor))

        cv2.imshow('Polytrack Vision', cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break 