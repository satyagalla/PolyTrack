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
        
        # 1. SETUP THE GAME WINDOW (Your code from yesterday)
        try:
            self.window = gw.getWindowsWithTitle('Edge')[0]
            self.window.activate()
        except Exception as e:
            print("Game window not found!")
            raise e

        # 2. DEFINE THE "HANDS" (Action Space)
        # Discrete(3) means 3 options: 0=Nothing, 1=Left, 2=Right
        # We can add 'W' (Forward) later or assume it's always pressed.
        self.action_space = spaces.Discrete(3)

        # 3. DEFINE THE "EYES" (Observation Space)
        # The AI expects a Box of numbers (Images).
        # Shape: (Height, Width, Channels) -> (84, 84, 1) for Grayscale
        # Low=0 (Black), High=255 (White), Type=uint8
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(84, 84, 1), dtype=np.uint8)

        # Initialize the screen capture tool
        self.sct = mss.mss()

    def step(self, action):
        # This is where the magic happens (Loop)
        # 1. Take Action (Press Key)
        # 2. Capture Screen (Get Observation)
        # 3. Calculate Reward (Did we crash?)
        # 4. Check Done (Game Over?)
        
        # Placeholder returns for now
        observation = np.zeros((84, 84, 1), dtype=np.uint8)
        reward = 1  # 1 point for surviving
        terminated = False
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Restart the game (Press 'R' or click Restart)
        super().reset(seed=seed)
        
        # Placeholder return
        observation = np.zeros((84, 84, 1), dtype=np.uint8)
        info = {}
        return observation, info

    def render(self):
        # Optional: Show what the bot sees
        pass

    def close(self):
        # Close connections
        pass

# TEST BLOCK (To make sure it runs)
if __name__ == "__main__":
    env = PolytrackEnv()
    print("Environment Created Successfully!")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")