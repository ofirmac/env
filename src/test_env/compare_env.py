#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Make sure your project root (with env/) is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.getcwd()))

from env.env_gym import GuestEnv

def reset_env(env, seed=None):
    try:
        return env.reset(seed=seed)
    except TypeError:
        obs = env.reset()
        return obs, {}

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, term, trunc, info = out
        return obs, reward, term or trunc, info
    else:
        obs, reward, done, info = out
        return obs, reward, done, info

def run_baseline(env, max_steps=200):
    obs, info = reset_env(env, seed=0)
    n = len(env.agent_params)
    ph_hist, g_hist = [], []
    for _ in range(max_steps):
        obs, reward, done, info = step_env(env, 0)  # always "wait"
        phonemes = np.array(info.get("phoneme", [0]*n), int)
        ph_hist.append(phonemes)
        g_hist.append(1.0 - info.get("env_reward", 0))
        if done: break
    return np.array(ph_hist), np.array(g_hist)

def run_old_solve(env, max_steps=200):
    # personality tweaks
    env.agent_params[0].update({'min_energy_to_speak':0.6,'energy_gain':0.01,'energy_decay':0.15,'max_speaking_time':2,'phonemes_per_step':1})
    env.agent_params[1].update({'min_energy_to_speak':0.3,'energy_gain':0.05,'energy_decay':0.10,'max_speaking_time':5,'phonemes_per_step':2})
    env.agent_params[2].update({'min_energy_to_speak':0.1,'energy_gain':0.10,'energy_decay':0.05,'max_speaking_time':8,'phonemes_per_step':4})

    obs, info = reset_env(env, seed=0)
    n = len(env.agent_params)
    cum = np.zeros(n, int)
    ph_hist, g_hist = [], []
    for _ in range(max_steps):
        target = int(np.argmin(cum))
        action = 5 + target
        obs, reward, done, info = step_env(env, action)
        phonemes = np.array(info.get("phoneme", [0]*n), int)
        cum += phonemes
        ph_hist.append(phonemes)
        g_hist.append(1.0 - info.get("env_reward", 0))
        if done: break
    return np.array(ph_hist), np.array(g_hist)

def run_new_solve(env, max_steps=200, window=10, threshold=5):
    # same personality tweaks
    env.agent_params[0].update({'min_energy_to_speak':0.6,'energy_gain':0.01,'energy_decay':0.15,'max_speaking_time':2,'phonemes_per_step':1})
    env.agent_params[1].update({'min_energy_to_speak':0.3,'energy_gain':0.05,'energy_decay':0.10,'max_speaking_time':5,'phonemes_per_step':2})
    env.agent_params[2].update({'min_energy_to_speak':0.1,'energy_gain':0.10,'energy_decay':0.05,'max_speaking_time':8,'phonemes_per_step':4})

    obs, info = reset_env(env, seed=0)
    n = len(env.agent_params)
    ph_hist, g_hist = [], []
    window_counts = np.zeros(n, int)
    cooldowns = np.zeros(n, int)
    from collections import deque
    dq = deque(maxlen=window)

    for _ in range(max_steps):
        # current window counts
        if dq:
            window_counts = np.sum(dq, axis=0)
        else:
            window_counts = np.zeros(n, int)

        quiet = int(np.argmin(window_counts))
        loud = int(np.argmax(window_counts))
        if window_counts[loud] - window_counts[quiet] > threshold:
            action = 1  # stop
        elif cooldowns[quiet] > 0:
            action = 0  # wait
        else:
            action = 5 + quiet
            cooldowns[quiet] = window

        obs, reward, done, info = step_env(env, action)
        phonemes = np.array(info.get("phoneme", [0]*n), int)
        ph_hist.append(phonemes)
        g_hist.append(1.0 - info.get("env_reward", 0))
        dq.append(phonemes)
        cooldowns = np.maximum(0, cooldowns - 1)
        if done: break

    return np.array(ph_hist), np.array(g_hist)

def plot_all(b_ph, b_g, o_ph, o_g, n_ph, n_g, out_dir="improve_plots"):
    os.makedirs(out_dir, exist_ok=True)
    T, N = b_ph.shape

    for i in range(N):
        plt.figure()
        plt.plot(b_ph[:,i], 'C0--', label='baseline')
        plt.plot(o_ph[:,i], 'C1-.', label='old solve')
        plt.plot(n_ph[:,i], 'C2-',  label='new solve')
        plt.title(f"Agent {i} phonemes")
        plt.xlabel("Step"); plt.ylabel("Phonemes")
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"agent_{i}.png"))
        plt.close()

    # Gini
    plt.figure()
    plt.plot(b_g, 'k--', label='baseline')
    plt.plot(o_g, 'k-.', label='old solve')
    plt.plot(n_g, 'k-',  label='new solve')
    plt.title("Gini index")
    plt.xlabel("Step"); plt.ylabel("Gini")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gini.png"))
    plt.close()

if __name__ == "__main__":
    STEPS = 200
    # Baseline
    env1 = GuestEnv()
    b_ph, b_g = run_baseline(env1, STEPS)

    # Old heuristic
    env2 = GuestEnv()
    o_ph, o_g = run_old_solve(env2, STEPS)

    # Improved heuristic
    env3 = GuestEnv()
    n_ph, n_g = run_new_solve(env3, STEPS, window=10, threshold=5)

    print("Baseline totals:", b_ph.sum(axis=0), "Gini:", round(b_g[-1],4))
    print("Old solve totals:", o_ph.sum(axis=0), "Gini:", round(o_g[-1],4))
    print("New solve totals:", n_ph.sum(axis=0), "Gini:", round(n_g[-1],4))

    plot_all(b_ph, b_g, o_ph, o_g, n_ph, n_g)
    print("Plots written to ./improve_plots/")
