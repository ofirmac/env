import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure env_gym.py is importable
sys.path.insert(0, os.path.abspath(os.getcwd()))
from guest_env.env_gym import GuestEnv

def safe_reset(env):
    """
    Try calling env.reset(); if it fails due to super().reset missing,
    manually initialize important fields and return obs, info.
    """
    try:
        return env.reset()
    except AttributeError:
        # Manual fallback initialization
        # Based on env_gym.py reset implementation
        if hasattr(env, 'seed') and env.seed is not None:
            env.rng = np.random.default_rng(env.seed)
        # energy initialization
        if getattr(env, 'energy_imbalance', 0) > 0:
            base = 0.5
            env.energy = np.array([
                base * (1 - env.energy_imbalance),
                base,
                base * (1 + env.energy_imbalance)
            ])
        else:
            env.energy = env.rng.uniform(0.4, 0.6, size=3)
        env.step_counter = 0
        env.gini_history = []
        env.phoneme_history = []
        # Reuse private method to get obs
        obs = env._get_obs()
        info = dict(
            num_of_step_env=env.step_counter,
            phoneme=env.phonemes.copy(),
            actions_stats=env.action_stats.copy(),
            env_reward=0.0,
            action_number=-1,
            gini_history=env.gini_history,
            phoneme_history=env.phoneme_history
        )
        return obs, info

def run_and_collect_phonemes(env, num_steps=200, action=0):
    obs, info = safe_reset(env)
    history = []
    for _ in range(num_steps):
        result = env.step(action)
        if len(result) == 5:
            _, _, terminated, truncated, info = result
            done = terminated or truncated
        else:
            _, _, done, info = result
        history.append(info['phoneme'])
        if done:
            break
    return np.array(history)

# 1) Either:
# env = GuestEnv(energy_imbalance=0.7)

# 2) Or, post‐init:
env = GuestEnv()
# make Agent 0 super‐reserved
env.agent_params[0].update({
    'min_energy_to_speak': 0.6,
    'energy_gain':         0.01,
    'energy_decay':        0.15,
    'max_speaking_time':   2,
    'phonemes_per_step':   1,
})

# make Agent 1 “normal”—leave at defaults (or tweak slightly)
env.agent_params[1].update({
    'min_energy_to_speak': 0.3,
    'energy_gain':         0.05,
    'energy_decay':        0.10,
    'max_speaking_time':   5,
    'phonemes_per_step':   2,
})

# make Agent 2 super‐talkative
env.agent_params[2].update({
    'min_energy_to_speak': 0.1,
    'energy_gain':         0.10,
    'energy_decay':        0.05,
    'max_speaking_time':   8,
    'phonemes_per_step':   4,
})

phonemes = run_and_collect_phonemes(env, num_steps=200, action=0)
T, N = phonemes.shape

# Create output dir
output_dir = './phoneme_plots'
os.makedirs(output_dir, exist_ok=True)
saved_files = []

# Save individual agent plots
for i in range(N):
    plt.figure()
    plt.plot(np.arange(T), phonemes[:, i])
    plt.title(f'Agent {i} Phoneme Count Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Phonemes')
    plt.grid(True)
    filepath = os.path.join(output_dir, f'agent_{i}.png')
    plt.savefig(filepath)
    plt.close()
    saved_files.append(filepath)

# Save combined plot
plt.figure()
for i in range(N):
    plt.plot(np.arange(T), phonemes[:, i], label=f'Agent {i}')
plt.title('All Agents Phoneme Counts Over Time')
plt.xlabel('Timestep')
plt.ylabel('Phonemes')
plt.legend()
plt.grid(True)
combined_path = os.path.join(output_dir, 'all_agents.png')
plt.savefig(combined_path)
plt.close()
saved_files.append(combined_path)

# Output saved file paths
for f in saved_files:
    print(f)
