import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import threading
import time
import pydirectinput
import pygetwindow as gw

# Import the new high-performance library
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

# --- WGC WRAPPER CLASS (Handles Async Capture) ---
class GameCapture:
    def __init__(self, partial_window_name="PolyTrack"):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.capture_thread = None
        self.partial_window_name = partial_window_name
        self.found_window = None

    def find_window(self):
        # Helper to find the exact window title (e.g., "PolyTrack - Chrome")
        print(f"Searching for window containing: '{self.partial_window_name}'...")
        possible_titles = gw.getAllTitles()
        for title in possible_titles:
            if self.partial_window_name in title and "Visual Studio" not in title and "VS Code" not in title:
                self.found_window = title
                print(f"✅ Found Game Window: '{self.found_window}'")
                return True
        print(f"❌ Error: Could not find any window with title '{self.partial_window_name}'")
        return False

    def start(self):
        if not self.find_window():
            raise RuntimeError("Game window not found. Open PolyTrack in your browser!")

        self.running = True
        self.capture_thread = threading.Thread(target=self._run_capture)
        self.capture_thread.start()
        
        # Block until the first frame arrives (prevents initial crash)
        print("Waiting for capture stream...")
        timeout = 0
        while self.latest_frame is None:
            time.sleep(0.1)
            timeout += 1
            if timeout > 50: # 5 seconds
                raise RuntimeError("Capture timed out. Ensure the game window is not minimized.")
        print("✅ Capture Stream Active!")

    def _run_capture(self):
        # Initialize WGC on the specific window
        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=self.found_window 
        )

        # Callback function for every new frame
        @capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            if not self.running:
                capture_control.stop()
                return

            # WGC provides a raw buffer. We convert it to numpy immediately.
            # Shape is (Height, Width, 4) -> BGRA
            img = np.frombuffer(frame.frame_buffer, dtype=np.uint8)
            img = img.reshape((frame.height, frame.width, 4))
            
            # Save it safely
            with self.lock:
                self.latest_frame = img

        @capture.event
        def on_closed():
            print("Capture Session Closed.")

        # Start the capture loop (blocking, so we run it in a thread)
        try:
            capture.start_free_threaded()
        except Exception as e:
            print(f"Capture crashed: {e}")

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()


# --- GYM ENVIRONMENT ---
class PolytrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(PolytrackEnv, self).__init__()
        self.render_mode = render_mode
        pydirectinput.PAUSE = 0.001 

        # 1. INITIALIZE CAPTURE
        self.cap = GameCapture(partial_window_name="PolyTrack")
        self.cap.start()

        # 2. ACTION SPACE (6 Actions: Gas/Brake Cluster)
        # 0=Gas, 1=Gas+L, 2=Gas+R, 3=Brake, 4=Brake+L, 5=Brake+R
        self.action_space = spaces.Discrete(6)

        # 3. OBSERVATION SPACE (84x84 Grayscale)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def step(self, action):
        # 1. Release Keys
        pydirectinput.keyUp('left'); pydirectinput.keyUp('right')
        pydirectinput.keyUp('up'); pydirectinput.keyUp('down')

        # 2. Press Keys (6-Action Logic)
        if action == 0:   pydirectinput.keyDown('up')
        elif action == 1: pydirectinput.keyDown('up'); pydirectinput.keyDown('left')
        elif action == 2: pydirectinput.keyDown('up'); pydirectinput.keyDown('right')
        elif action == 3: pydirectinput.keyDown('down')
        elif action == 4: pydirectinput.keyDown('down'); pydirectinput.keyDown('left')
        elif action == 5: pydirectinput.keyDown('down'); pydirectinput.keyDown('right')

        # 3. Get Frame (Instant from Memory)
        raw_screen = self.cap.get_latest_frame()
        
        # Fallback if frame is empty
        if raw_screen is None: 
            raw_screen = np.zeros((84, 84, 3), dtype=np.uint8)

        # Convert BGRA to Grayscale (Drop Alpha channel)
        gray_frame = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2GRAY)
        
        terminated = self._check_crash(gray_frame)
        
        resized_obs = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        final_obs = np.expand_dims(resized_obs, axis=-1)

        # 4. Reward Function (Phase 3)
        if terminated:
            reward = -20.0
        else:
            reward = 0.1
            if action == 0: # Gas Bonus
                reward += 0.05 

        return final_obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pydirectinput.keyUp('up'); pydirectinput.keyUp('down')
        pydirectinput.keyUp('left'); pydirectinput.keyUp('right')
        
        # HARD RESET ('t')
        pydirectinput.press('t') 
        time.sleep(0.5) 
        
        raw_screen = self.cap.get_latest_frame()
        if raw_screen is None: raw_screen = np.zeros((100, 100, 3), dtype=np.uint8)
        
        gray_frame = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2GRAY)
        resized_obs = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        final_obs = np.expand_dims(resized_obs, axis=-1)
        return final_obs, {}

    def _check_crash(self, gray_frame):
        # ROI: Optimized for Full Screen or Windowed
        h, w = gray_frame.shape
        # Look at the "Sky" area where the "Press R to Restart" text appears
        roi = gray_frame[int(h*0.12):int(h*0.4), int(w*0.25):int(w*0.75)]
        _, thresh = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(thresh) > 100

    def close(self):
        self.cap.stop()

if __name__ == "__main__":
    try:
        env = PolytrackEnv()
        env.reset()
        print("\n------------------------------------------------")
        print("✅ SPEED TEST MODE (Visualizer OFF)")
        print("------------------------------------------------")
        
        start = time.time()
        frames = 0
        while True:
            # Step the environment (Capture + Process)
            obs, _, done, _, _ = env.step(0)
            frames += 1
            
            # --- COMMENTED OUT THE VISUALIZER ---
            # cv2.imshow("Bot Vision", ...) 
            # if cv2.waitKey(1) == ord('q'): break
            
            if done: env.reset()
            
            # Print FPS every 1 second
            if time.time() - start > 1.0:
                print(f"🚀 TRUE FPS: {frames}")
                frames = 0
                start = time.time()
                
    except Exception as e:
        print(f"\n❌ FAILED: {e}")