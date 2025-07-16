# GuestEnv Documentation

> **Purpose**
> `GuestEnv` simulates a moderated three‑person conversation where a **meta‑agent** (the RL “Guest”) intervenes with *wait*, *stop*, *stare\_at*, and *encourage* actions to balance participation.  The environment measures equality with the **Gini index** of accumulated phonemes, rewarding the agent for fair turn‑taking while penalising over‑intervention.

---

## 1. Key Concepts

| Concept              | Meaning                                                                                            | Where Stored            |
| -------------------- | -------------------------------------------------------------------------------------------------- | ----------------------- |
| **Energy**           | Per‑speaker drive to talk (0‑1). Increases passively and via agent actions; decays while speaking. | `self.energy[3]`        |
| **Speaking Time**    | Consecutive steps the current speaker has held the floor.                                          | `self.speaking_time[3]` |
| **Phonemes**         | Cumulative speech tokens per speaker. ‑Used to compute Gini.                                       | `self.phonemes[3]`      |
| **Current Speaker**  | `-1` when no‑one speaks, else 0‑2.                                                                 | `self.current_speaker`  |
| **Imbalance Factor** | Bias that makes higher‑index speakers naturally gain energy faster (simulates extroversion).       | `self.imbalance_factor` |
| **Energy Imbalance** | Initial handicap so one speaker starts with more/less energy.                                      | `self.energy_imbalance` |
| **Reward Shaping**   | Optional bonus/penalty terms that speed up learning.                                               | `self.reward_shaping`   |

---

## 2. Spaces

* **Action space** `Discrete(8)`

  | Id  | Guest command             |
  | --- | ------------------------- |
  | 0   | `wait` (repeat last)      |
  | 1   | `stop`                    |
  | 2‑4 | `stare_at speaker` (0‑2)  |
  | 5‑7 | `encourage speaker` (0‑2) |

* **Observation space** `Box(0,1, shape=(17,))`

  1. 3 × energy
  2. 3 × speaking‑time ratio
  3. 3 × phoneme proportion (softmax)
  4. 4‑way one‑hot **current speaker** (`nobody + 3 speakers`)
  5. Gini index (scalar)
  6. Phoneme std & range (2)
  7. Progress (scalar)

---

## 3. Episode Lifecycle – High‑Level Flow

```mermaid
graph TD
    A[Agent selects action a_t] --> B[env.step(a_t)]
    B --> C[Internal state update]
    C --> D[Compute reward r_t]
    D --> E[Generate obs_{t+1}]
    E --> F{Done?}
    F -- no --> A
    F -- yes --> G[env.reset()]
```

---

## 4. reset() Logic

```mermaid
flowchart TD
    R0[reset(seed, options)] --> R1[Seed RNG]
    R1 --> R2[Initialise Energy]
    R2 --> R3[Zero speaking_time & phonemes]
    R3 --> R4[current_speaker = -1]
    R4 --> R5[step_counter = 0]\n action_stats = 0
    R5 --> R6[_get_obs()]
    R6 --> R7[return (obs, info)]
```

### Highlights

* **Energy initialisation**

  * If `energy_imbalance>0`: deterministic asymmetric vector.
  * Else: uniform \[0.4, 0.6].
* Histories (`gini_history`, `phoneme_history`) cleared.

---

## 5. step() Logic

```mermaid
flowchart TD
    S0[step(action)] --> S1[step_counter++  & action_stats[action]++]
    S1 --> S2{Guest action?}
    S2 -- stop (1) --> S2a[current_speaker=-1 & zero energy]
    S2 -- stare_at (2‑4) --> S2b[boost target energy]
    S2 -- encourage (5‑7) --> S2c[boost if floor free]
    S2 -- wait (0) --> S2d[no direct change]
    S2d --> S3[Passive energy dynamics per speaker]
    S2a --> S3
    S2b --> S3
    S2c --> S3
    S3 --> S4{Speaker present?}
    S4 -- none --> S4a[Find candidate speakers≥threshold & pick max energy]
    S4 -- yes --> S4b[Increase speaking_time; stop if limits crossed]
    S4a --> S5[Update phonemes if someone speaking]
    S4b --> S5
    S5 --> S6[reward = 1‑gini ± shaping]
    S6 --> S7[Build obs, info]
    S7 --> S8{step_counter≥max_steps?}
    S8 -- yes --> term
    S8 -- no --> cont
```

---

## 6. Function‑by‑Function Reference

### `__init__(max_steps=600, seed=42, imbalance_factor=0.0, energy_imbalance=0.0, reward_shaping=True)`

Initialises spaces, RNG, agent‑specific parameters, and tracking buffers.  **Agent parameters** (`self.agent_params`) tune each speaker’s personality: min energy to start, decay/gain per step, max sentence length, phonemes generated each step.

### `_get_obs()`

Constructs the 17‑dimensional observation described in §2, normalising where required to keep the box bounded in \[0, 1] (except std/range which are ≤1 by design).

### `_gini()`

Pure function computing Gini index of `self.phonemes` – zero when equal, one when only one speaker has spoken.

### `reset()`

Implements flow in §4, returning `(obs, info)` where `info` includes step counter, phoneme vector, empty action stats, etc.

### `step(action)`

Implements detailed flow in §5.  Returns `(obs, reward, terminated, truncated, info)` compatible with the Gymnasium API.

| Returned flag | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| `terminated`  | Episode ended because `step_counter ≥ max_steps`.    |
| `truncated`   | Always `False` (placeholder for future time‑limits). |

---

## 7. Usage Example

```python
import gymnasium as gym
from stable_baselines3 import PPO
from env_gym import GuestEnv

env = gym.wrappers.RecordEpisodeStatistics(
          GuestEnv(max_steps=600, imbalance_factor=0.2, reward_shaping=True))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

**Tips**

* Wrap with `DummyVecEnv` or `SubprocVecEnv` for parallel training.
* Hyperparameters that matter most: learning‑rate schedule, reward γ, and clip‑range.
* Turn off `reward_shaping` after convergence for cleaner evaluation.

---

## 8. Extending the Environment

* **Different group sizes** – change all arrays to dynamic length and enlarge action space.
* **Alternative fairness metrics** – swap `_gini()` with e.g. Jain’s fairness or entropy.
* **Richer actions** – add “transfer energy” or “mute” by appending to `ACTIONS` and updating `step()`.

---

## 9. Troubleshooting

| Symptom            | Possible Cause                      | Fix                                                 |
| ------------------ | ----------------------------------- | --------------------------------------------------- |
| Agent never speaks | `imbalance_factor` too high         | Lower to <0.5                                       |
| Guest spams `stop` | Reward shaping: big bonus for stops | Reduce shaping coefficients                         |
| Training unstable  | Sparse rewards                      | Increase `reward_shaping`, frame‑stack observations |

---

## 10. File Map

```text
env_gym.py  – Environment + (optional) training script
tb_logs/     – TensorBoard data (if you kept the training part)
```
