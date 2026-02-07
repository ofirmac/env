## GuestEnv Environment Deep Documentation

This document provides a comprehensive, in-depth overview of the `GuestEnv` Gymnasium environment defined in `env_gym.py`. It covers the purpose, internal state, dynamics, methods, and usage of the environment.

---

### 1. Overview

* **Purpose**: Simulates a conversation between three agents (participants) moderated by a special "Guest" controller. The goal is to equalize speaking opportunities and measure conversational balance.
* **Key Entities**:

  * **Agents (0, 1, 2)**: Each has distinct energy thresholds and speaking characteristics.
  * **Guest**: An external controller (your RL policy) that can influence conversation by taking one of eight actions.

---

### 2. Constants & Imports

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Action index → human-readable mapping
actions = {
    0: "wait",
    1: "stop",
    2: "stare_at 0",
    3: "stare_at 1",
    4: "stare_at 2",
    5: "encourage 0",
    6: "encourage 1",
    7: "encourage 2",
}
```

---

### 3. Initialization (`__init__`)

**Signature**:

```python
def __init__(
    self,
    *,
    max_steps: int = 600,
    seed: int = 42,
    imbalance_factor: float = 0.0,
    energy_imbalance: float = 0.0
):
```

* **Parameters**:

  * `max_steps`: (int) Maximum number of steps in one episode before truncation.
  * `seed`: (int): Random number generator seed for reproducibility.
  * `imbalance_factor` (float, 0.0–1.0): Biases how energy gains and losses are applied across agents. A value of 0.0 yields symmetric energy dynamics; higher values favor agent indices 2 over 1 and 0.
  * `energy_imbalance` (float, 0.0–1.0): Sets initial skew in agents’ starting energy levels. 0.0 means all energies are sampled uniformly; positive values shift the initial energy distribution so that higher-indexed agents start with more energy.
* **Spaces**:

  * `action_space = Discrete(8)`
  * `observation_space = Box(low=0.0, high=[1.0,1.0,∞]×3, dtype=float32)`

* **Agent Parameters**:

  ```python
  self.agent_params = {
      0: { 'min_energy_to_speak': 0.4, 'energy_decay': 0.08, 'energy_gain': 0.04, 'max_speaking_time': 4, 'phonemes_per_step': 2 },
      1: { 'min_energy_to_speak': 0.3, 'energy_decay': 0.10, 'energy_gain': 0.05, 'max_speaking_time': 5, 'phonemes_per_step': 1 },
      2: { 'min_energy_to_speak': 0.25,'energy_decay': 0.12,'energy_gain': 0.06,'max_speaking_time': 6,'phonemes_per_step': 1 },
  }
  ```

**Agent Parameter Definitions** (contained in `self.agent_params`): Each agent (0, 1, 2) is governed by a set of sub-parameters:

- **`min_energy_to_speak`** (`float`):  
  The minimum energy threshold an agent must have to be eligible to start speaking. If an agent’s energy falls below this value, the agent cannot become (or remain) the speaker.

- **`energy_decay`** (`float`):  
  The amount by which an agent’s energy is reduced on each step while that agent is speaking. Ensures that prolonged speaking eventually depletes the agent’s ability to continue speaking.

- **`energy_gain`** (`float`):  
  The amount by which an agent’s energy increases on each step when that agent is not speaking. Allows idle agents to regain capacity to speak over time.

- **`max_speaking_time`** (`int`):  
  The maximum number of consecutive steps an agent can speak before being forced to yield. Once an agent’s `speaking_time` reaches this integer, speaking ends even if the agent still has energy.

- **`phonemes_per_step`** (`int`):  
  The number of phoneme units credited to an agent’s phonemes counter on each step that the agent is speaking. Used to calculate fairness (via Gini) and drive reward shaping.



* **Internal State Variables**:

  * `energy`: \[agent0, agent1, agent2] (ndarray of shape (3,)): Current energy level for each agent, ranging [0.0, 1.0]. Reflects how much capacity an agent has to speak: higher energy increases the chance an agent will become (or remain) the speaker..
  * `speaking_time`: (ndarray of shape (3,)): Number of consecutive steps each agent has been speaking in the current turn.
  * `phonemes`: (ndarray of shape (3,)): Cumulative count of phonemes (speech units) each agent has produced in the episode.
  * `current_speaker`: (int): Index of the agent currently speaking, or -1 if none.
  * `step_counter`: (int): Count of steps taken so far in the episode.
  * `action_stats`: (ndarray of length 8): Counts of how many times each Guest action has been invoked.
  * `gini_history`, `phoneme_history`: l(lists): Track the history of Gini coefficients and phoneme arrays for each step, for logging or analysis.

---

### 4. Reset (`reset`)

**Behavior**:

1. Re-seed RNG if provided.
2. Initialize `energy`:

   * If `energy_imbalance > 0`: `[0.5*(1-imbalance), 0.5, 0.5*(1+imbalance)]`.
   * Else: uniform random in \[0.4, 0.6].
3. Zero out `speaking_time`, `phonemes`, `current_speaker`, `step_counter`, `action_stats`.
4. Clear history lists.
5. Return initial observation array and an `info` dict containing:

   * `num_of_step_env`, `phoneme` (zeros), `actions_stats` (zeros),
   * `env_reward = 0.0`, `action_number = -1`, plus empty histories.

```python
obs, info = env.reset()
```

---

### 5. Observation (`_get_obs`)

Constructs a 9-element vector:

```python
obs = [
    energy[0], speaking_time[0]/max_speaking_time[0], phonemes[0],
    energy[1], speaking_time[1]/max_speaking_time[1], phonemes[1],
    energy[2], speaking_time[2]/max_speaking_time[2], phonemes[2],
]
```

Normalized values help continuous control.

---

### 6. Gini Coefficient (`_gini`)

Measures inequality of phoneme distribution:

```python
total = sum(phonemes)
if total == 0: return 0.0
x = phonemes.astype(float)
diffs = |x_i - x_j| summed over all pairs
gini = diffs / (2 * n * total)
```

A lower Gini ⇒ more balanced speaking.

---

### 7. Step (`step`)

**Signature**:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

1. **Action Accounting**:

   * Increment `step_counter` and `action_stats[action]`.

2. **Guest Interventions** (`imbalance_factor` scaling):

   * `action == 0` (wait): No immediate effect.
   * `action == 1` (stop): If someone is speaking, zero their energy and stop them.
   * `2–4` (stare\_at i): Boost `energy[i]` by `0.2*(1-imbalance_factor)`.
   * `5–7` (encourage i): If no one is speaking, boost `energy[i]` by `0.3*(1-imbalance_factor)`.

3. **Agent Energy Dynamics**:

   ```python
   for each agent i:
       if i != current_speaker:
           energy[i] += energy_gain[i] * (1 + imbalance_factor*(i-1))
       else:
           energy[i] -= energy_decay[i]
       clip energy[i] to [0,1]
   ```

   * Agents off-camera regain energy; the active speaker loses energy.
   * `imbalance_factor` biases gains: favors higher-index agents when >0.

4. **Speaking Dynamics**:

   * **If no one speaking**:

     * Candidates = agents with `energy >= min_energy_to_speak`.
     * Choose agent with max energy → `current_speaker` = that agent; reset its `speaking_time`.

   * **Else (agent continues)**:

     * Increment `speaking_time[current_speaker]`.
     * If `speaking_time >= max_speaking_time` *or* `energy < min_energy_to_speak`: agent yields (`current_speaker = -1`).

5. **Phoneme Counting**:

   * If someone is speaking, add `phonemes_per_step` to their count.

6. **Reward Computation**:

   ```python
   base_reward = 1 - gini()
   if someone just started speaking (speaking_time == 1): +0.2 bonus
   if someone is nearing end of max_speaking_time (>80%): -0.1 penalty
   reward = base_reward ± shaping
   ```

7. **Termination**:

   * `terminated` always `False`.
   * `truncated` = `(step_counter >= max_steps)`.

8. **Return**:

   * New `obs`, scalar `reward`, `terminated`, `truncated` flags, and `info` dict containing metrics and histories.

---

### 8. Info Dictionary

Keys returned at each step:

* `num_of_step_env`: current time step
* `phoneme`: array of phoneme counts
* `actions_stats`: counts of each Guest action
* `env_reward`: raw equality-based reward (1 − Gini)
* `action_number`: index of the action just taken
* `gini_history`: list of past Gini values
* `phoneme_history`: list of past phoneme arrays

Use these to monitor training progress or for custom logging.

---

### 9. Usage Example

```python
import gymnasium as gym
from env_gym import GuestEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create vectorized env
env = DummyVecEnv([lambda: GuestEnv(max_steps=300, imbalance_factor=0.2)])
# Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

### 10. Tuning Tips

* **`imbalance_factor`**:

  * 0.0 ⇒ fair natural dynamics.
  * ↑ ⇒ agents 2 favored, agent 0 penalized → increases imbalance.

* **`energy_imbalance`**:

  * 0.0 ⇒ random start.
  * ↑ ⇒ more skew in initial energies.

* **Reward Shaping**:

  * Modify bonus/penalty values to encourage smoother turn-taking.

* **Max Steps**:

  * Adjust episode length to match desired conversation duration.

---

This documentation should equip you to fully understand and customize `GuestEnv` for research or training purposes.
