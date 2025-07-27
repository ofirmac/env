# guest_rl_train.py
# ──────────────────────────────────────────────────────────────
# Requirements:
#   pip install gymnasium stable-baselines3 torch matplotlib
# ──────────────────────────────────────────────────────────────

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env.env_gym import GuestEnv

# ──────────────────────────────────────────────────────────────
# 1) SETUP LOG DIRS
# ──────────────────────────────────────────────────────────────
ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("logs", ts)
TB_DIR  = os.path.join(LOG_DIR, "tensorboard")
PLOT_DIR= os.path.join(LOG_DIR, "plots")
os.makedirs(TB_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"[TRAIN] Logs → {LOG_DIR}")
print(f"[TRAIN] TensorBoard → {TB_DIR}")

# ──────────────────────────────────────────────────────────────
# 2) ENV FACTORY (no changes to env_gym.py)
# ──────────────────────────────────────────────────────────────
def make_env():
    env = GuestEnv(max_steps=5000, reward_shaping=False)
    env.agent_params[0].update({'min_energy_to_speak':0.6,'energy_gain':0.01,'energy_decay':0.15,'max_speaking_time':2,'phonemes_per_step':1})
    env.agent_params[1].update({'min_energy_to_speak':0.3,'energy_gain':0.05,'energy_decay':0.10,'max_speaking_time':5,'phonemes_per_step':2})
    env.agent_params[2].update({'min_energy_to_speak':0.1,'energy_gain':0.10,'energy_decay':0.05,'max_speaking_time':8,'phonemes_per_step':4})
    return Monitor(env)         # records per-episode rewards/lengths

env = DummyVecEnv([make_env])

# ──────────────────────────────────────────────────────────────
# 3) CALLBACK TO LOG & PLOT EPISODES
# ──────────────────────────────────────────────────────────────
class EpisodeLogger(BaseCallback):
    def __init__(self, plot_dir: str, verbose=0):
        super().__init__(verbose)
        self.plot_dir = plot_dir
        self.ep_rewards = []
        self.ep_ginis   = []

    def _on_step(self) -> bool:
        # In a VecEnv, infos is a list
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][idx]
                # 1) Episode reward from Monitor
                ep_rew = info.get("episode", {}).get("r")
                if ep_rew is not None:
                    self.ep_rewards.append(ep_rew)
                # 2) Avg‐Gini: access env.gini_history
                env = self.training_env.envs[idx]
                if hasattr(env, "gini_history") and env.gini_history:
                    avg_gini = np.mean(env.gini_history)
                else:
                    avg_gini = 0.0
                self.ep_ginis.append(avg_gini)
                # 3) Push scalars to TensorBoard
                self.logger.record("episode/reward", ep_rew)
                self.logger.record("episode/avg_gini", avg_gini)
        return True

    def _on_training_end(self) -> None:
        # Plot Reward
        plt.figure()
        plt.plot(self.ep_rewards, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Per-Episode Reward")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "reward.png"))
        plt.close()

        # Plot Avg Gini
        plt.figure()
        plt.plot(self.ep_ginis, label="Avg Gini", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Gini")
        plt.title("Per-Episode Avg-Gini")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "gini.png"))
        plt.close()

# ──────────────────────────────────────────────────────────────
# 4) CREATE & TRAIN THE MODEL
# ──────────────────────────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=TB_DIR,
    batch_size=64,
    n_steps=2048,
    learning_rate=3e-4,
    clip_range=0.2,
)

callback = EpisodeLogger(plot_dir=PLOT_DIR)
model.learn(total_timesteps=200_000, callback=callback)

print(f"[TRAIN] Done! Plots saved in {PLOT_DIR}")
print(f"[TRAIN] You can view TensorBoard with:\n    tensorboard --logdir {TB_DIR}")
