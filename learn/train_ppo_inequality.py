import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env.env_gym import GuestEnv

class MetricsCallback(BaseCallback):
    """Callback for recording Gini coefficient and phoneme counts."""
    def __init__(self):
        super().__init__()
        self.gini = []
        self.phonemes = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        self.gini.append(info.get("gini_history", [0])[-1])
        self.phonemes.append(info.get("phoneme_history", [[0,0,0]])[-1])
        return True


def train(total_timesteps: int = 20000, results_dir: str = "ppo_results") -> None:
    os.makedirs(results_dir, exist_ok=True)

    env = DummyVecEnv([
        lambda: GuestEnv(max_steps=600, imbalance_factor=0.3, energy_imbalance=0.5)
    ])
    model = PPO("MlpPolicy", env, verbose=1)
    callback = MetricsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(results_dir, "ppo_guest"))

    phonemes = np.array(callback.phonemes)

    plt.figure(figsize=(8, 4))
    plt.plot(callback.gini)
    plt.xlabel("step")
    plt.ylabel("Gini coefficient")
    plt.title("Conversation inequality during training")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gini.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    for idx in range(phonemes.shape[1]):
        plt.plot(phonemes[:, idx], label=f"Agent {idx}")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("Phonemes")
    plt.title("Phoneme count per agent")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phonemes.png"))
    plt.close()


if __name__ == "__main__":
    train()
