import os
import sys
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
from datetime import datetime

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from guest_env.env_gym import GuestEnv

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = []
        self.observation_space = gym.spaces.Box(
            low=np.repeat(self.observation_space.low, n_frames, axis=0),
            high=np.repeat(self.observation_space.high, n_frames, axis=0),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.n_frames
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        self.frames.pop(0)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.concatenate(self.frames, axis=0)

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.gini_history = []
        self.reward_history = []
        self.phoneme_history = []

    def _on_step(self):
        info = self.locals['infos'][0]
        self.gini_history.append(info['gini_history'][-1] if info['gini_history'] else 0)
        self.reward_history.append(self.locals['rewards'][0])
        self.phoneme_history.append(info['phoneme_history'][-1] if info['phoneme_history'] else [0, 0, 0])
        return True

def plot_training_results(callback, save_path):
    plt.figure(figsize=(15, 10))
    
    # Plot Gini coefficient
    plt.subplot(2, 2, 1)
    plt.plot(callback.gini_history)
    plt.title('Gini Coefficient Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Gini Coefficient')
    
    # Plot rewards
    plt.subplot(2, 2, 2)
    plt.plot(callback.reward_history)
    plt.title('Rewards Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    
    # Plot phoneme distribution
    plt.subplot(2, 2, 3)
    phonemes = np.array(callback.phoneme_history)
    plt.plot(phonemes[:, 0], label='Participant 0')
    plt.plot(phonemes[:, 1], label='Participant 1')
    plt.plot(phonemes[:, 2], label='Participant 2')
    plt.title('Phoneme Distribution Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Phonemes')
    plt.legend()
    
    # Plot final phoneme distribution
    plt.subplot(2, 2, 4)
    final_phonemes = phonemes[-1]
    plt.bar(['P0', 'P1', 'P2'], final_phonemes)
    plt.title('Final Phoneme Distribution')
    plt.ylabel('Phonemes')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(env, model_class, model_kwargs, total_timesteps, save_path):
    model = model_class(env=env, **model_kwargs)
    callback = MetricsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    return callback

def main():
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Environment parameters
    env_params = {
        'max_steps': 1000,
        'seed': 42,
        'imbalance_factor': 0.2,  # Slight imbalance
        'energy_imbalance': 0.1   # Slight initial energy imbalance
    }

    # Training parameters
    total_timesteps = 1_000_000
    n_frames = 10  # For frame stacking

    # Algorithms to compare
    algorithms = {
        'PPO': {
            'class': PPO,
            'kwargs': {
                'policy': 'MlpPolicy',
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'verbose': 1
            }
        },
        'RecurrentPPO': {
            'class': RecurrentPPO,
            'kwargs': {
                'policy': 'MlpLstmPolicy',
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'verbose': 1
            }
        },
        'DQN': {
            'class': DQN,
            'kwargs': {
                'policy': 'MlpPolicy',
                'learning_rate': 1e-4,
                'buffer_size': 100000,
                'learning_starts': 10000,
                'batch_size': 64,
                'tau': 1.0,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 1000,
                'exploration_fraction': 0.1,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'verbose': 1
            }
        }
    }

    # Train each algorithm with and without frame stacking
    for algo_name, algo_config in algorithms.items():
        print(f"\nTraining {algo_name}...")
        
        # Without frame stacking
        env = DummyVecEnv([lambda: GuestEnv(**env_params)])
        callback = train_model(
            env, 
            algo_config['class'], 
            algo_config['kwargs'],
            total_timesteps,
            f"{results_dir}/{algo_name}_model"
        )
        plot_training_results(
            callback,
            f"{results_dir}/{algo_name}_results.png"
        )

        # With frame stacking
        env = DummyVecEnv([lambda: GuestEnv(**env_params)])
        env = VecFrameStack(env, n_stack=n_frames)
        callback = train_model(
            env,
            algo_config['class'],
            algo_config['kwargs'],
            total_timesteps,
            f"{results_dir}/{algo_name}_stacked_model"
        )
        plot_training_results(
            callback,
            f"{results_dir}/{algo_name}_stacked_results.png"
        )

if __name__ == "__main__":
    main() 
