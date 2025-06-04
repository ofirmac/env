#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import the environment
sys.path.insert(0, os.path.abspath(os.getcwd()))
# 1) Adjust this import to wherever your env lives:
from guest_env.env_gym import GuestEnv

def reset_env(env, seed=None):
    """
    Reset fallback that works with both Gym >=0.27 and older.
    Returns (obs, info).
    """
    try:
        return env.reset(seed=seed)
    except TypeError:
        # older gym returns obs only
        obs = env.reset()
        return obs, {}

def step_env(env, action):
    """
    Step fallback that works with both Gym >=0.27 and older.
    Returns (obs, reward, done, info).
    """
    out = env.step(action)
    if len(out) == 5:
        obs, reward, term, trunc, info = out
        return obs, reward, term or trunc, info
    else:
        obs, reward, done, info = out
        return obs, reward, done, info

def solve_env(max_steps=200):
    env = GuestEnv()
    env.agent_params[0].update({
        'min_energy_to_speak': 0.6,
        'energy_gain':         0.01,
        'energy_decay':        0.15,
        'max_speaking_time':   2,
        'phonemes_per_step':   1,
    })
    env.agent_params[1].update({
        'min_energy_to_speak': 0.3,
        'energy_gain':         0.05,
        'energy_decay':        0.10,
        'max_speaking_time':   5,
        'phonemes_per_step':   2,
    })
    env.agent_params[2].update({
        'min_energy_to_speak': 0.1,
        'energy_gain':         0.10,
        'energy_decay':        0.05,
        'max_speaking_time':   8,
        'phonemes_per_step':   4,
})

    obs, info = reset_env(env, seed=0)
    n_agents = 3
    cum_phonemes = np.zeros(n_agents, dtype=int)
    phoneme_hist = []
    gini_hist = []

    for t in range(max_steps):
        # 1) choose the quietest agent so far
        target = int(np.argmin(cum_phonemes))
        action = 5 + target   # actions 5,6,7 are "encourage 0/1/2"

        # 2) step
        obs, reward, done, info = step_env(env, action)

        # 3) record
        phonemes = np.array(info.get("phoneme", [0]*n_agents), dtype=int)
        cum_phonemes += phonemes
        phoneme_hist.append(phonemes)
        # Gini is stored as history of (1 − reward)
        gini_hist.append(1.0 - info.get("env_reward", 0))

        if done:
            break

    return np.array(phoneme_hist), np.array(gini_hist), cum_phonemes

def plot_results(phoneme_hist, gini_hist, out_dir="solve_plots"):
    os.makedirs(out_dir, exist_ok=True)
    T, N = phoneme_hist.shape

    # per-agent phonemes
    for i in range(N):
        plt.figure()
        plt.plot(range(T), phoneme_hist[:, i], lw=2)
        plt.title(f"Agent {i} phonemes over time (solved)")
        plt.xlabel("Step")
        plt.ylabel("Phonemes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"agent_{i}_solve.png"))
        plt.close()

    # combined
    plt.figure()
    for i in range(N):
        plt.plot(range(T), phoneme_hist[:, i], label=f"Agent {i}")
    plt.title("All agents phonemes over time (solved)")
    plt.xlabel("Step")
    plt.ylabel("Phonemes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_agents_solve.png"))
    plt.close()

    # Gini
    plt.figure()
    plt.plot(range(len(gini_hist)), gini_hist, lw=2, color="C3")
    plt.title("Gini Index over time (solved)")
    plt.xlabel("Step")
    plt.ylabel("Gini")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gini_solve.png"))
    plt.close()

if __name__ == "__main__":
    phoneme_hist, gini_hist, cum = solve_env(max_steps=200)
    print("Final cumulative phonemes:", cum)
    print("Final Gini:", gini_hist[-1] if len(gini_hist) else None)
    plot_results(phoneme_hist, gini_hist)
    print("Plots written to ./solve_plots")
