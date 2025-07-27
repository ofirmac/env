# guest_env_softmax.py  (drop-in for guest_rl_train.py)
# -----------------------------------------------------
# from __future__ import annotations
import argparse
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import matplotlib.pyplot as plt

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# ---------------------------------------------------------------------------
# Moderator actions
ACTIONS = {
    0: "wait",        1: "stop",
    2: "stare_at 0",  3: "stare_at 1",  4: "stare_at 2",
    5: "encourage 0", 6: "encourage 1", 7: "encourage 2",
}
ENCOURAGE_ACTIONS = {5, 6, 7}

# ---------------------------------------------------------------------------
# Soft-max parameters  ✨ NEW ✨
ALPHA, BETA, GAMMA = 0.50, 0.35, 0.15      # talk-habit / engagement / fairness
TAU                = 0.40                  # temperature (↓ deterministic)
MAX_STREAK         = 200                   # how long before “fairness” saturates
ENG_DECAY          = 0.01                  # engagement –0.01 every step
ENG_BOOST          = 0.20                  # encourage gives +0.20 engagement

# ---------------------------------------------------------------------------
class GuestEnv(gym.Env):
    """
    GuestEnv with soft-max speaker-selection.
    Reward, utterance length, obs space remain unchanged.
    """
    metadata = {"render_modes": []}

    def __init__(self, *, max_steps: int = 600, seed: int = 42,
                 imbalance_factor: float = 0.0,
                 energy_imbalance: float = 0.0,
                 reward_shaping: bool = True,
                 logfile: bool = False):
        super().__init__()
        self.max_steps       = max_steps
        self.rng             = np.random.default_rng(seed)
        self.imbalance_factor = imbalance_factor
        self.energy_imbalance = energy_imbalance
        self.reward_shaping   = reward_shaping

        # ------- NEW soft-max state ----------------------------------- #
        self.talk_prior     = 0.3 + 0.6 * self.rng.beta(2, 2, size=3)  # α
        self.engagement     = np.full(3, 0.5, dtype=np.float32)        # β
        self.silence_streak = np.zeros(3, dtype=np.int32)              # γ

        # --------------------------------------------------------------- #
        self.action_space      = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(17,), dtype=np.float32)

        # --- (unchanged) agent parameters, state arrays, logging vars --
        self.agent_params = {
            0: dict(min_energy_to_speak=0.5, energy_decay=0.05,
                    energy_gain=0.04, max_speaking_time=5, phonemes_per_step=3),
            1: dict(min_energy_to_speak=0.5, energy_decay=0.50,
                    energy_gain=0.05, max_speaking_time=5, phonemes_per_step=3),
            2: dict(min_energy_to_speak=0.9, energy_decay=0.20,
                    energy_gain=0.06, max_speaking_time=5, phonemes_per_step=3),
        }
        self.energy         = np.zeros(3)
        self.speaking_time  = np.zeros(3)
        self.phonemes       = np.zeros(3, dtype=int)
        self.current_speaker = -1
        self.step_counter    = 0
        self.action_stats    = np.zeros(len(ACTIONS), dtype=int)
        self.gini_history, self.phoneme_history, self.env_reward = [], [], []

        if logfile:
            logger.add("guest_{time:YYYY-MM-DD}.log", enqueue=True)

    # =============================  OBSERVATION  (unchanged) ============= #
    def _get_obs(self) -> np.ndarray:
        obs = []
        # 1 energy
        obs.extend(self.energy)
        # 2 speaking-ratios
        obs.extend([min(1.0, self.speaking_time[i] /
                        self.agent_params[i]['max_speaking_time']) for i in range(3)])
        # 3 phoneme distribution
        total_p = np.sum(self.phonemes)
        obs.extend(self.phonemes / total_p if total_p else np.ones(3)/3)
        # 4 current speaker one-hot
        speaker_enc = np.zeros(4); speaker_enc[(self.current_speaker+1)] = 1
        obs.extend(speaker_enc)
        # 5 gini
        obs.append(self._gini())
        # 6 phoneme std / range
        if total_p:
            phon_std = np.std(self.phonemes) / (total_p/3)
            phon_rng = (np.max(self.phonemes) - np.min(self.phonemes)) / (total_p/3)
        else:
            phon_std = phon_rng = 0.0
        obs.extend([phon_std, phon_rng])
        # 7 progress
        obs.append(self.step_counter / self.max_steps)
        return np.asarray(obs, dtype=np.float32)

    # =============================  GINI (unchanged) ===================== #
    def _gini(self) -> float:
        total = np.sum(self.phonemes)
        if total == 0: return 0.0
        diffs = np.abs(self.phonemes[:, None] - self.phonemes).sum()
        return diffs / (2 * len(self.phonemes) * total)

    # =============================  RESET  =============================== #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # energy init
        if self.energy_imbalance:
            base = 0.5
            self.energy = np.array([
                base*(1-self.energy_imbalance), base, base*(1+self.energy_imbalance)])
        else:
            self.energy = self.rng.uniform(0.4, 0.6, 3)

        # clear dynamic vars
        self.speaking_time[:] = 0
        self.phonemes[:]      = 0
        self.current_speaker  = -1
        self.step_counter     = 0
        self.action_stats[:]  = 0
        self.gini_history.clear()
        self.phoneme_history.clear()
        # soft-max trackers
        self.engagement[:]    = 0.5
        self.silence_streak[:] = 0

        return self._get_obs(), {}

    # =============================  STEP  ================================ #
    def step(self, action: int):
        self.step_counter += 1
        self.action_stats[action] += 1

        # -------- 1. moderator action effects on ENERGY & ENGAGEMENT ---- #
        if action == 1 and self.current_speaker != -1:            # stop
            self.energy[self.current_speaker] = 0.0
            self.current_speaker = -1
        elif 2 <= action <= 4:                                    # stare_at
            tgt = action - 2
            self.energy[tgt] = min(1.0, self.energy[tgt] + 0.2*(1-self.imbalance_factor))
        elif action in ENCOURAGE_ACTIONS:                         # encourage
            tgt = action - 5
            if self.current_speaker == -1:
                self.energy[tgt] = min(1.0, self.energy[tgt] + 0.3*(1-self.imbalance_factor))
                self.engagement[tgt] = min(1.0, self.engagement[tgt] + ENG_BOOST)

        # -------- 2. update engagement decay  -------------------------- #
        self.engagement = np.maximum(0.0, self.engagement - ENG_DECAY)

        # -------- 3. energy natural dynamics  -------------------------- #
        for i in range(3):
            p = self.agent_params[i]
            if i != self.current_speaker:
                gain = p['energy_gain'] * (1 + self.imbalance_factor*(i-1))
                self.energy[i] = min(1.0, self.energy[i] + gain)
            else:
                self.energy[i] = max(0.0, self.energy[i] - p['energy_decay'])

        # -------- 4. select speaker (✨ soft-max) ----------------------- #
        if self.current_speaker == -1:
            # candidates who satisfy min_energy
            cands = [i for i in range(3) if self.energy[i] >=
                     self.agent_params[i]['min_energy_to_speak']]
            if cands:
                # scores only for candidates, −inf for others
                scores = np.full(3, -np.inf)
                streak_norm = np.clip(self.silence_streak / MAX_STREAK, 0, 1)
                scores_cand = (ALPHA*self.talk_prior[cands] +
                               BETA *self.engagement[cands] +
                               GAMMA*streak_norm[cands])
                scores[cands] = scores_cand
                probs = np.exp(scores/TAU); probs /= probs.sum()
                self.current_speaker = self.rng.choice(3, p=probs)
                self.speaking_time[self.current_speaker] = 0
                # reset fairness timer
                self.silence_streak[self.current_speaker] = 0
        else:
            p = self.agent_params[self.current_speaker]
            self.speaking_time[self.current_speaker] += 1
            if (self.speaking_time[self.current_speaker] >= p['max_speaking_time'] or
                    self.energy[self.current_speaker] < p['min_energy_to_speak']):
                self.current_speaker = -1

        # -------- 5. silence streak update ----------------------------- #
        self.silence_streak += 1
        if self.current_speaker != -1:
            self.silence_streak[self.current_speaker] = 0

        # -------- 6. phoneme accumulation (unchanged) ------------------ #
        if self.current_speaker != -1:
            inc = self.agent_params[self.current_speaker]['phonemes_per_step']
            self.phonemes[self.current_speaker] += inc

        # -------- 7. reward  (unchanged) ------------------------------- #
        gini   = self._gini()
        reward = 1.0 - gini
        if self.reward_shaping and self.current_speaker != -1:
            if self.speaking_time[self.current_speaker] == 1:
                reward += 0.2
            p = self.agent_params[self.current_speaker]
            if self.speaking_time[self.current_speaker] > p['max_speaking_time']*0.8:
                reward -= 0.1

        terminated = False
        truncated  = self.step_counter >= self.max_steps
        obs = self._get_obs()

        # history (for plotting / logging)
        self.gini_history.append(gini)
        self.phoneme_history.append(self.phonemes.copy())
        self.env_reward.append(reward)

        info = dict(num_of_step_env=self.step_counter,
                    phoneme=self.phonemes.copy(),
                    actions_stats=self.action_stats.copy(),
                    env_reward=reward,
                    action_number=int(action))
        return obs, reward, terminated, truncated, info

# =======================================================================
# -----------------------  Quick demo & plot  ---------------------------
# =======================================================================
# def demo(steps: int = 1000, seed: int = 0):
#     env = GuestEnv(max_steps=steps, seed=seed)
#     env.agent_params[0].update({'min_energy_to_speak':0.6,'energy_gain':0.01,'energy_decay':0.15,'max_speaking_time':2,'phonemes_per_step':1})
#     env.agent_params[1].update({'min_energy_to_speak':0.3,'energy_gain':0.05,'energy_decay':0.10,'max_speaking_time':5,'phonemes_per_step':2})
#     env.agent_params[2].update({'min_energy_to_speak':0.1,'energy_gain':0.10,'energy_decay':0.05,'max_speaking_time':8,'phonemes_per_step':4})

#     obs, _ = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         obs, r, term, trunc, _ = env.step(action)
#         done = term or trunc

#     totals = env.phonemes
#     plot_phoneme_history(env.phoneme_history, title="baseline - Phonemes")


# def plot_phoneme_history(phoneme_history: list[np.ndarray],
#                          title: str = "baseline - Phonemes") -> None:
#     """
#     Draw cumulative phoneme counts for the three agents across the episode.

#     Parameters
#     ----------
#     phoneme_history : list[np.ndarray]
#         The list you already append to in GuestEnv.step(); each element is a
#         3-element array of total phonemes at that step.
#     title : str
#         Figure title.
#     """
#     # Convert to (T,3) ndarray
#     data  = np.asarray(phoneme_history)           # shape = (steps, 3)
#     steps = np.arange(len(data))

#     plt.figure(figsize=(8, 4))
#     plt.plot(steps, data[:, 0], label="Agent 0")  # blue
#     plt.plot(steps, data[:, 1], label="Agent 1")  # orange
#     plt.plot(steps, data[:, 2], label="Agent 2")  # green
#     plt.title(title, fontsize=14, pad=10)
#     plt.xlabel("Step")
#     plt.ylabel("Phonemes")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--steps", type=int, default=200,
#                     help="Episode length for demo run & plot")
#     ap.add_argument("--seed", type=int, default=0)
#     args = ap.parse_args()
#     demo(args.steps, args.seed)
