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

    # 2. DEFINE MODEL
    model = PPO(
        "CnnPolicy", 
        env,
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=0.0001,    # Reduced from 0.0003 (Stabilizes the Wiggle)
        n_steps=2048,
        batch_size=128,          # Increased from 64 (Smoother updates)
        ent_coef=0.01            # Encourages exploration of new buttons (like Brake)
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