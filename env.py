import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pygetwindow as gw
import pydirectinput
import time

class PolytrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    # --- CONFIGURATION ---
    # Crop offsets to isolate the game view (adjust if needed)
    CROP_OFFSET = {
        "top": 110,    
        "left": 15,    
        "right": 15,   
        "bottom": 120  
    }

    def __init__(self, render_mode=None):
        super(PolytrackEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # 1. OPTIMIZATION: Zero-latency inputs
        pydirectinput.PAUSE = 0.001 

        # 2. ROBUST WINDOW SETUP
        self.window = None
        possible_titles = ['PolyTrack', 'Edge', 'Chrome', 'Firefox', 'Brave']
        
        print("Searching for game window...")
        for title in possible_titles:
            wins = gw.getWindowsWithTitle(title)
            if wins:
                self.window = max(wins, key=lambda w: w.width * w.height)
                print(f"Attached to window: '{self.window.title}'")
                break
        
        if not self.window:
            raise RuntimeError("Game window not found! Please open PolyTrack.")

        try:
            self.window.activate()
        except:
            pass
        time.sleep(0.5)

        # 3. EXPANDED ACTION SPACE (6 Actions)
        # 0: Max Acceleration (W)
        # 1: Soft Left (W + A)
        # 2: Soft Right (W + D)
        # 3: Hard Brake (S)
        # 4: Hard Left (S + A) <--- NEW: Rotates car while slowing
        # 5: Hard Right (S + D) <--- NEW: Rotates car while slowing
        self.action_space = spaces.Discrete(6)

        # 4. OBSERVATION SPACE (84x84 Grayscale)
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(84, 84, 1), dtype=np.uint8)

        self.sct = mss.mss()
        self._update_monitor()

    def _update_monitor(self):
        self.monitor = {
            "top": self.window.top + self.CROP_OFFSET["top"], 
            "left": self.window.left + self.CROP_OFFSET["left"], 
            "width": self.window.width - (self.CROP_OFFSET["left"] + self.CROP_OFFSET["right"]), 
            "height": self.window.height - (self.CROP_OFFSET["top"] + self.CROP_OFFSET["bottom"])
        }

    def step(self, action):
        # Reset keys...
        pydirectinput.keyUp('left'); pydirectinput.keyUp('right')
        pydirectinput.keyUp('up'); pydirectinput.keyUp('down')

        if action == 0:   # GAS STRAIGHT
            pydirectinput.keyDown('up')
        elif action == 1: # GAS LEFT
            pydirectinput.keyDown('up'); pydirectinput.keyDown('left')
        elif action == 2: # GAS RIGHT
            pydirectinput.keyDown('up'); pydirectinput.keyDown('right')
        elif action == 3: # BRAKE STRAIGHT
            pydirectinput.keyDown('down')
        elif action == 4: # BRAKE LEFT (The Cornering Fix)
            pydirectinput.keyDown('down'); pydirectinput.keyDown('left')
        elif action == 5: # BRAKE RIGHT (The Cornering Fix)
            pydirectinput.keyDown('down'); pydirectinput.keyDown('right')
        
        # ... rest of function ...

        # 2. CAPTURE STATE
        raw_screen = np.array(self.sct.grab(self.monitor))
        gray_frame = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2GRAY)
        
        # 3. CHECK TERMINATION
        terminated = self._check_crash(gray_frame)

        # 4. PROCESS OBSERVATION
        resized_obs = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        final_obs = np.expand_dims(resized_obs, axis=-1)

        # 5. REWARD STRATEGY (Simple Curriculum)
        if terminated:
            reward = -10.0  # As requested
        else:
            reward = 0.1    # Survival Incentive

        return final_obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Release all keys
        pydirectinput.keyUp('up')
        pydirectinput.keyUp('down')
        pydirectinput.keyUp('left')
        pydirectinput.keyUp('right')

        # Restart Level
        pydirectinput.press('r')
        time.sleep(0.2) 
        
        # Initial Frame
        raw_screen = np.array(self.sct.grab(self.monitor))
        gray_frame = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2GRAY)
        resized_obs = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        final_obs = np.expand_dims(resized_obs, axis=-1)

        return final_obs, {}

    def _check_crash(self, gray_frame):
        # ROI: Top Center (Look for "Press R" or "Level Failed")
        h, w = gray_frame.shape
        roi = gray_frame[:int(h*0.2), int(w*0.25):int(w*0.75)]
        
        _, thresh = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)

        return white_pixels > 50

    def close(self):
        pydirectinput.keyUp('up')
        pydirectinput.keyUp('down')
        pydirectinput.keyUp('left')
        pydirectinput.keyUp('right')
        self.sct.close()

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    env = PolytrackEnv(render_mode="human")
    obs, _ = env.reset()
    print("Environment Loaded. Press 'q' to quit.")
    try:
        while True:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            
            big_obs = cv2.resize(obs, (420, 420), interpolation=cv2.INTER_NEAREST)
            if done:
                cv2.putText(big_obs, "CRASH", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                env.reset()
                
            cv2.imshow("Agent View", big_obs)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        env.close()
        cv2.destroyAllWindows()