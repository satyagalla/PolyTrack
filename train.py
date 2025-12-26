import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env import PolytrackEnv 

# CONFIG
MODELS_DIR = "models/PPO"
LOG_DIR = "logs"
TIMESTEPS = 20000 

if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

def main():
    # 1. INIT ENV
    env = PolytrackEnv()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    # Stack 4 frames so the AI can "feel" speed
    env = VecFrameStack(env, n_stack=4)

    # 2. DEFINE MODEL (Auto-Resume Logic)
    model_path = f"{MODELS_DIR}/100000.zip" # Example: Change this to your latest save if restarting
    
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new model...")
        model = PPO(
            "CnnPolicy", 
            env,
            verbose=1, 
            tensorboard_log=LOG_DIR,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=128,
            ent_coef=0.01,
            device="cuda" # Force GPU usage
        )

    print("--- STARTING TRAINING ---")
    print("Please click the PolyTrack window within 5 seconds.")
    time.sleep(5) 

    # 3. LOOP
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{MODELS_DIR}/{TIMESTEPS*iters}")
        print(f"Saved Checkpoint: {iters}")

if __name__ == "__main__":
    main()