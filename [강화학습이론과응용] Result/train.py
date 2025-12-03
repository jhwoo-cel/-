# train.py
import os
import time

from grid_amr_env import AMRSmallEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class TimeSummaryCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.last_step_time = None
        self.total_sampling_time = 0.0

        self.total_learning_time = 0.0
        self.rollout_count = 0


    def _on_rollout_start(self):
        self.rollout_start_time = time.time()


    def _on_rollout_end(self):
        learning_time = time.time() - self.rollout_start_time
        self.total_learning_time += learning_time
        self.rollout_count += 1


    def _on_step(self):
        now = time.time()
        if self.last_step_time is not None:
            self.total_sampling_time += (now - self.last_step_time)
        self.last_step_time = now
        return True

    def _on_training_end(self):
        print("\n=================== TRAIN SUMMARY ===================")
        print(f"Total Sampling Time: {self.total_sampling_time:.4f} sec")
        if self.num_timesteps > 0:
            print(f"Avg Sampling Time per Step: {self.total_sampling_time / self.num_timesteps:.6f} sec")

        print(f"\nTotal Learning Time: {self.total_learning_time:.4f} sec")
        if self.rollout_count > 0:
            print(f"Avg Learning Time per Rollout: {self.total_learning_time / self.rollout_count:.6f} sec")
        print("======================================================\n")


log_dir = f"./tensorboard_logs/run_{time.strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(log_dir, exist_ok=True)


def make_env():
    env = AMRSmallEnv()
    env = Monitor(env, log_dir, info_keywords=("is_success",))
    return env


env = DummyVecEnv([make_env])

new_logger = configure(log_dir, ["stdout", "tensorboard"])


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=1024,
    batch_size=128,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    device="cuda",
)

model.set_logger(new_logger)

callback = TimeSummaryCallback(verbose=1)


model.learn(total_timesteps=1_000_000, callback=callback)
model.save("multi_amr_ppo")

print("[INFO] Training Complete!")
print(f"[INFO] TensorBoard log dir: {log_dir}")
print('Run: tensorboard --logdir tensorboard_logs')
