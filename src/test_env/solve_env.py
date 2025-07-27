import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import the environment
sys.path.insert(0, os.path.abspath(os.getcwd()))
from env.env_gym import GuestEnv

def safe_reset(env, seed=None, options=None):
    """
    Reset fallback, replicating GuestEnv.reset without calling super().reset.
    """
    try:
        return env.reset(seed=seed, options=options)
    except AttributeError:
        if seed is not None:
            env.seed = seed
            env.rng = np.random.default_rng(seed)
        # Initialize energies
        if env.energy_imbalance > 0:
            base_energy = 0.5
            env.energy = np.array([
                base_energy * (1 - env.energy_imbalance),
                base_energy,
                base_energy * (1 + env.energy_imbalance)
            ])
        else:
            env.energy = env.rng.uniform(0.4, 0.6, size=3)
        # Reset internal state
        env.speaking_time = np.zeros(3)
        env.phonemes = np.zeros(3, dtype=int)
        env.current_speaker = -1
        env.step_counter = 0
        env.action_stats[:] = 0
        env.gini_history = []
        env.phoneme_history = []
        # Return initial obs and info
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
        res = env.step(action)
        if len(res) == 5:
            _, _, terminated, truncated, info = res
            done = terminated or truncated
        else:
            _, _, done, info = res
        history.append(info['phoneme'])
        if done:
            break
    return np.array(history)

# Instantiate the environment
env = GuestEnv()

# Configure agent personalities
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

# Run dry-run
phonemes = run_and_collect_phonemes(env, num_steps=200, action=0)
T, N = phonemes.shape

# Prepare output
output_dir = './phoneme_plots_config'
os.makedirs(output_dir, exist_ok=True)
saved_files = []

# Separate plots
for i in range(N):
    plt.figure()
    plt.plot(np.arange(T), phonemes[:, i])
    plt.title(f'Agent {i} Phoneme Count (Configured)')
    plt.xlabel('Timestep')
    plt.ylabel('Phonemes')
    plt.grid(True)
    fpath = os.path.join(output_dir, f'agent_{i}_cfg.png')
    plt.savefig(fpath)
    plt.close()
    saved_files.append(fpath)

# Combined plot
plt.figure()
for i in range(N):
    plt.plot(np.arange(T), phonemes[:, i], label=f'Agent {i}')
plt.title('All Agents Phonemes (Configured)')
plt.xlabel('Timestep')
plt.ylabel('Phonemes')
plt.legend()
plt.grid(True)
combined = os.path.join(output_dir, 'all_agents_cfg.png')
plt.savefig(combined)
plt.close()
saved_files.append(combined)

print("Generated plots:")
for f in saved_files:
    print(f)
