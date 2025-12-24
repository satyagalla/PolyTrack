import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env import PolytrackEnv
import cv2

# --- CONFIGURATION ---
# Update this filename to match your latest saved model in the models/PPO folder
# Based on your logs, it's likely around 1000000.zip
MODEL_PATH = "models/PPO/960000" 

def main():
    print(f"Loading model from: {MODEL_PATH}")
    
    # 1. SETUP ENV (Must match train.py EXACTLY)
    env = PolytrackEnv(render_mode="human")
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4) # <--- Critical!

    # 2. LOAD MODEL
    # We load the zip file. We don't need to specify policy/learning_rate here.
    try:
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"Error: Could not find model file '{MODEL_PATH}.zip'")
        print("Check your 'models/PPO' folder and update the MODEL_PATH variable.")
        return

    print("-----------------------------------------")
    print("STARTING INFERENCE RUN")
    print("Click the game window immediately!")
    print("Press 'q' in the python window to quit.")
    print("-----------------------------------------")
    time.sleep(3) # Time to switch focus

    # 3. RUN LOOP
    obs = env.reset()
    while True:
        # deterministic=True means "Pick the BEST action", not a random one.
        # We want to see its peak performance, not exploration.
        action, _states = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, info = env.step(action)
        
        # Optional: Render what the agent sees to confirm inputs
        # We need to grab the last frame from the stack to visualize
        # obs shape is (1, 84, 84, 4). We want (84, 84).
        current_frame = obs[0, :, :, -1] 
        big_obs = cv2.resize(current_frame, (420, 420), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Agent Brain", big_obs)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()