# test.py
from grid_amr_env import AMRSmallEnv
from stable_baselines3 import PPO

env = AMRSmallEnv(render_delay=0.2)   
model = PPO.load("multi_amr_ppo")

obs, _ = env.reset()
terminated, truncated = False, False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

input("Press Enter to exit")



