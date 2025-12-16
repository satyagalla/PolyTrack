import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pygetwindow as gw
import pydirectinput
import time

class PolytrackEnv(gym.Env):
    def __init__(self):
        super(PolytrackEnv, self).__init__()
        
        # 1. SETUP WINDOW
        try:
            self.window = gw.getWindowsWithTitle('Edge')[0] # Changed to Edge as per your code
            self.window.activate()
        except Exception as e:
            print("Game window not found!")
            raise e

        # 2. ACTION SPACE (4 Discrete Actions)
        # 0=Forward, 1=Left, 2=Right, 3=Brake
        self.action_space = spaces.Discrete(4)

        # 3. OBSERVATION SPACE (84x84 Grayscale)
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(84, 84, 1), dtype=np.uint8)

        self.sct = mss.mss()

    def step(self, action):
        # 1. RESET STEERING/BRAKE
        # We release Left/Right/Down every frame to ensure we don't get stuck turning
        pydirectinput.keyUp('left')
        pydirectinput.keyUp('right')
        pydirectinput.keyUp('down')

        # 2. HANDLE GAS vs. BRAKE (The "Always Forward" Logic)
        if action == 3: # Brake Mode
            pydirectinput.keyUp('up')     # Release Gas
            pydirectinput.keyDown('down') # Press Brake
        else: # Driving Mode (0, 1, 2)
            pydirectinput.keyDown('up')   # GAS IS ALWAYS ON
            
            # 3. HANDLE STEERING
            if action == 1: # Left
                pydirectinput.keyDown('left')
            elif action == 2: # Right
                pydirectinput.keyDown('right')
            # Action 0 is implicitly "Straight + Gas"

        # Wait a tiny bit for physics
        time.sleep(0.05)
        
        # --- PHASE 2: OBSERVATION ---
        observation = self.get_observation()
        
        # --- PHASE 3: TERMINATION ---
        terminated = self.is_game_over()
        
        # --- PHASE 4: REWARD ---
        if terminated:
            reward = -10
        else:
            reward = 1 

        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Press 'R' to restart the level
        pydirectinput.press('r')
        time.sleep(0.5) # Wait for level to reload
        
        observation = self.get_observation()
        info = {}
        return observation, info

    def get_observation(self):
        # Update monitor in case window moved
        monitor = {
            "top": self.window.top + 105, 
            "left": self.window.left + 15, 
            "width": self.window.width - 30, 
            "height": self.window.height - 120
        }
        
        # Capture
        img = np.array(self.sct.grab(monitor))
        
        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84))
        final_obs = np.expand_dims(resized, axis=2)
        
        return final_obs

    def is_game_over(self):
        # 1. Grab the current screen area
        monitor = {
            "top": self.window.top + 105, 
            "left": self.window.left + 15, 
            "width": self.window.width - 30, 
            "height": self.window.height - 120
        }
        img = np.array(self.sct.grab(monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # 2. Define the "Watch Box" (Bottom Center where "Press R" appears)
        # You might need to tune these numbers!
        height, width = gray.shape
        # Look at the top 20% of the screen
        roi = gray[:int(height*0.2), int(width*0.25):int(width*0.75)]
        
        # 3. Check for Bright White Text
        # If the pixels are very bright (white text > 240), we assume crash text is there.
        # We check if there is a significant amount of white pixels.
        _, thresh = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        
        # If we see more than 50 white pixels in that area, assume it's text.
        if white_pixels > 50:
            return True
        return False
    
if __name__ == "__main__":
    env = PolytrackEnv()
    
    obs, info = env.reset()
    for i in range(500): # Increased to 500 frames so you have time to watch
        action = env.action_space.sample()
        
        obs, reward, terminated, _, _ = env.step(action)
        
        # --- FIX FOR HUMAN EYES ---
        # The AI sees 'obs' (84x84). We create 'big_obs' just for you.
        # Scale it up 5x (to 420x420) using Nearest Neighbor (keeps pixels sharp)
        big_obs = cv2.resize(obs, (420, 420), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Bot View (Magnified)", big_obs)
        
        # IF TERMINATED, PRINT IT (Debug your Death Detector)
        if terminated:
            print("CRASH DETECTED!")
            env.reset()

        if cv2.waitKey(1) == ord('q'):
            break
            
    cv2.destroyAllWindows()