# guest_rl_train.py
# -----------------------------------------------------------
# pip install gymnasium stable-baselines3 torch tensorboard
# -----------------------------------------------------------

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import sys
import random
from typing import Optional

logger.remove()
logger.add(
    sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
    level="INFO",
)

# ------------------------------------------------------------------------
# 1.  ENVIRONMENT
# ------------------------------------------------------------------------

ACTIONS = {
    0: "wait",
    1: "stare_at 0",
    2: "stare_at 1",
    3: "stare_at 2",
    4: "encourage 0",
    5: "encourage 1",
    6: "encourage 2",
}


class GuestEnv(gym.Env):
    """
    A conversation environment with 3 participants and one Guest moderator.
    Each agent has unique personality traits that affect their speaking behavior:
    - Agent 0: Reserved and thoughtful (speaks less but more meaningfully)
    - Agent 1: Balanced and moderate (standard speaking pattern)
    - Agent 2: Energetic and talkative (speaks more frequently)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_steps: int = 600,
        seed: int = 42,
        imbalance_factor: float = 0.0,  # 0.0 to 1.0, controls natural imbalance
        energy_imbalance: float = 0.0,
        reward_shaping=True,
        logfile=False,
        encourage_base_effect: float = 0.9,
        encourage_duration_steps: int = 10,
        encourage_stack: bool = True,
    ):  # 0.0 to 1.0, controls initial energy imbalance
        super().__init__()
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.imbalance_factor = imbalance_factor  # It gives you a single knob (imbalance_factor) to favor “higher-index” agents (2) over “lower-index” ones (0), with agent 1 in the middle.
        self.energy_imbalance = energy_imbalance
        self.reward_shaping = reward_shaping

        # Action space: 8 discrete Guest actions
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation space: [energy, speaking_time, total_phonemes] * 3
        # high = np.array([1.0, 1.0, np.inf] * 3, dtype=np.float32)
        # self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # temporary buff
        self.encourage_base_effect = float(encourage_base_effect)
        self.encourage_duration_steps = int(encourage_duration_steps)
        self.encourage_stack = bool(encourage_stack)

        # active buffs per speaker: list[dict(amount, remaining)]
        self._encourage_buffs = [[] for _ in range(3)]

        # Agent-specific parameters
        self.agent_params = {
            0: {  # Reserved agent
                "min_energy_to_speak": 0.5,
                "energy_decay": 0.05,
                "energy_gain": 0.04,
                "max_speaking_time": 5,
                "phonemes_per_step": 3,
            },
            1: {  # Balanced agent
                "min_energy_to_speak": 0.5,
                "energy_decay": 0.5,
                "energy_gain": 0.05,
                "max_speaking_time": 5,
                "phonemes_per_step": 3,
            },
            2: {  # Energetic agent
                "min_energy_to_speak": 0.9,
                "energy_decay": 0.2,
                "energy_gain": 0.06,
                "max_speaking_time": 5,
                "phonemes_per_step": 3,
            },
        }

        # State variables
        self.energy = np.zeros(3)
        self.speaking_time = np.zeros(3)
        self.phonemes = np.zeros(3, dtype=int)
        self.current_speaker = -1
        self.step_counter = 0
        self.action_stats = np.zeros(len(ACTIONS), dtype=int)

        # Turn-taking selection config
        self.speaker_tau = (
            3.0  # temperature: 0<tau; lower = more greedy, higher = more random
        )
        self.selection_mode = "softmax"  # keep for future modes ("uniform", "greedy"), currently using softmax

        # Imbalance tracking
        self.gini_history = []
        self.env_reward = []
        self.phoneme_history = []
        # === SPEAK-style selection (energy-as-baseline) ===
        self.speak_d: float = 0.79   # immediate boost after speaking
        self.speak_b: float = 0.79   # exponential decay rate by missed turns
        self.speak_alpha: float = 0.50  # mix: alpha*baseline_energy + (1-alpha)*momentary
        self.speak_eps: float = 0.00    # optional ε-exploration (0 = off)
        self.num_agents = 3
        # per-agent “missed since last spoke” counters
        self.speak_skipped = np.zeros(self.num_agents, dtype=int)

        # if you track current speaker; use -1 when none
        self.current_speaker: int = getattr(self, "current_speaker", -1)
        if logfile:
            logger.add("guest_{time:YYYY-MM-DD}.log", enqueue=True)

    def _get_obs(self) -> np.ndarray:
        """Enhanced observation with softmax distributions and relative features."""
        obs = []

        # 1. ENERGY LEVELS (normalized)
        energy_normalized = self.energy.copy()
        obs.extend(energy_normalized)  # [3 values: 0-1]

        # 2. SPEAKING TIME RATIOS (normalized by max)
        speaking_ratios = []
        for i in range(3):
            ratio = self.speaking_time[i] / self.agent_params[i]["max_speaking_time"]
            speaking_ratios.append(min(1.0, ratio))
        obs.extend(speaking_ratios)  # [3 values: 0-1]

        # 3. PHONEME DISTRIBUTION (softmax normalized)
        total_phonemes = np.sum(self.phonemes)
        if total_phonemes > 0:
            phoneme_distribution = (
                self.phonemes / total_phonemes
            )  # Relative proportions
        else:
            phoneme_distribution = np.ones(3) / 3  # Equal if no speech yet
        obs.extend(phoneme_distribution)  # [3 values: sum=1.0]

        # 4. CURRENT SPEAKER (one-hot encoded)
        speaker_encoding = np.zeros(4)  # [nobody, agent0, agent1, agent2]
        if self.current_speaker == -1:
            speaker_encoding[0] = 1.0
        else:
            speaker_encoding[self.current_speaker + 1] = 1.0
        obs.extend(speaker_encoding)  # [4 values: one-hot]

        # 5. BALANCE METRICS
        gini = self._gini()
        obs.append(gini)  # [1 value: 0-1, lower=better]

        # 6. PHONEME STATISTICS (normalized)
        if total_phonemes > 0:
            phoneme_std = np.std(self.phonemes) / (total_phonemes / 3)  # Normalized std
            phoneme_range = (np.max(self.phonemes) - np.min(self.phonemes)) / (
                total_phonemes / 3
            )
        else:
            phoneme_std = 0.0
            phoneme_range = 0.0
        obs.extend([phoneme_std, phoneme_range])  # [2 values: balance measures]

        # 7. PROGRESS INDICATOR
        progress = self.step_counter / self.max_steps
        obs.append(progress)  # [1 value: 0-1]

        return np.asarray(obs, dtype=np.float32)  # Total: 18 features

    def _gini(self) -> float:
        total = np.sum(self.phonemes)
        if total == 0:
            return 0.0
        x = self.phonemes.astype(float)
        diffs = np.abs(x[:, None] - x[None, :]).sum()
        n = len(x)
        return float(diffs / (2 * n * total))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        # Initialize with imbalanced energy levels if specified
        if self.energy_imbalance > 0:
            base_energy = 0.5
            self.energy = np.array(
                [
                    base_energy * (1 - self.energy_imbalance),
                    base_energy,
                    base_energy * (1 + self.energy_imbalance),
                ]
            )
        else:
            self.energy = self.rng.uniform(0.4, 0.6, size=3)

        self.speaking_time = np.zeros(3)
        self.phonemes = np.zeros(3, dtype=int)
        self.current_speaker = -1
        self.step_counter = 0
        self.action_stats[:] = 0
        self.gini_history = []
        self.phoneme_history = []
        self._encourage_buffs = [[] for _ in range(3)]

        obs = self._get_obs()
        info = dict(
            num_of_step_env=self.step_counter,
            phoneme=self.phonemes.copy(),
            actions_stats=self.action_stats.copy(),
            env_reward=0.0,
            action_number=-1,
            gini_history=self.gini_history,
            phoneme_history=self.phoneme_history,
        )
        return obs, info

    @logger.catch
    def step(self, action: int):
        self.step_counter += 1
        self.action_stats[action] += 1

        # Process Guest action with imbalance factor
        logger.debug(f"{self.current_speaker=}")
        if 1 <= action <= 3:  # stare_at
            target = action - 1
            effect = 0.1 * (1 - self.imbalance_factor)
            self.energy[target] = min(1.0, self.energy[target] + effect)
        elif 4 <= action <= 6:  # encourage
            target = action - 4
            self._apply_encourage(target)
            # if self.current_speaker == -1:
            #     effect = 0.5 * (1 - self.imbalance_factor)
            #     self.energy[target] = min(1.0, self.energy[target] + effect)

        # Agent-specific energy dynamics
        for i in range(3):
            params = self.agent_params[i]
            if i != self.current_speaker:
                gain = params["energy_gain"] * (1 + self.imbalance_factor * (i - 1))
                self.energy[i] = min(1.0, self.energy[i] + gain)
                logger.debug(f"{self.energy[i]}")
            else:
                self.energy[i] = max(0.0, self.energy[i] - params["energy_decay"])
                logger.debug(f"{self.energy[i]}")

        # Speaking dynamics with agent-specific parameters
        if self.current_speaker == -1:
            # Find potential speakers based on their individual thresholds
            candidates = []
            for i in range(3):
                if self.energy[i] >= self.agent_params[i]["min_energy_to_speak"]:
                    candidates.append(i)

            if len(candidates) > 0:
                probs = self._speak_probs()
                chosen = self._sample_from_probs(probs)   # returns an int
                # Choose speaker with highest energy
                c = self._random_sample(self.energy)
                self.current_speaker = c
                # self.current_speaker = candidates[np.argmax(self.energy[candidates])]
                logger.debug(f"{self.current_speaker=}")
                self.speaking_time[self.current_speaker] = 0
        else:
            # Current speaker continues or yields based on their parameters
            params = self.agent_params[self.current_speaker]
            self.speaking_time[self.current_speaker] += 1
            if (
                self.speaking_time[self.current_speaker] >= params["max_speaking_time"]
                or self.energy[self.current_speaker] < params["min_energy_to_speak"]
            ):
                self.current_speaker = -1

        # Update phonemes with agent-specific rates
        if self.current_speaker != -1:
            self.phonemes[self.current_speaker] += self.agent_params[
                self.current_speaker
            ]["phonemes_per_step"]

        # Calculate reward
        gini = self._gini()
        reward = 1.0 - gini  # Base reward on equality

        if self.reward_shaping:
            # Additional reward shaping
            if self.current_speaker != -1:
                # Bonus for new speaker
                if self.speaking_time[self.current_speaker] == 1:
                    reward += 0.2
                # Penalty for long speaking turns
                if (
                    self.speaking_time[self.current_speaker]
                    > self.agent_params[self.current_speaker]["max_speaking_time"] * 0.8
                ):
                    reward -= 0.1

        terminated = False
        truncated = self.step_counter >= self.max_steps

        obs = self._get_obs()
        info = dict(
            num_of_step_env=self.step_counter,
            phoneme=self.phonemes.copy(),
            actions_stats=self.action_stats.copy(),
            env_reward=1.0 - gini,
            action_number=int(action),
            gini_history=self.gini_history,
            phoneme_history=self.phoneme_history,
            energy=self.energy,
        )

        # Track history
        self.gini_history.append(1.0 - info["env_reward"])
        self.env_reward.append(info["env_reward"])
        self.phoneme_history.append(self.phonemes.copy())
        self._tick_encourage_buffs()
        return obs, reward, terminated, truncated, info

    # ---- NEW helper methods somewhere in the class ----

    def _apply_encourage(self, target: int) -> None:
        """Apply a temporary encourage buff to `target`."""
        # if self.current_speaker != -1:
        #     return  # your existing logic: only when nobody is speaking
        # base effect scaled by imbalance, then cap to headroom so we can remove it later safely
        effect = self.encourage_base_effect * (1 - self.imbalance_factor)
        headroom = 1.0 - float(self.energy[target])
        delta = max(0.0, min(effect, headroom))
        if delta <= 0.0:
            return

        # apply now
        self.energy[target] = min(1.0, float(self.energy[target]) + delta)

        # record the buff so we can roll it back after N steps
        buff = {"amount": float(delta), "remaining": int(self.encourage_duration_steps)}

        if self.encourage_stack:
            self._encourage_buffs[target].append(buff)
        else:
            # replace any existing buff: first remove them, then set a single new one
            for b in self._encourage_buffs[target]:
                self.energy[target] = max(
                    0.0, min(1.0, float(self.energy[target]) - b["amount"])
                )
            self._encourage_buffs[target] = [buff]

    def _tick_encourage_buffs(self) -> None:
        """Decrement timers and remove expired buff amounts from energy."""
        for i in range(3):
            kept = []
            for b in self._encourage_buffs[i]:
                b["remaining"] -= 1
                if b["remaining"] <= 0:
                    # expire: remove exactly what we added
                    self.energy[i] = max(
                        0.0, min(1.0, float(self.energy[i]) - b["amount"])
                    )
                else:
                    kept.append(b)
            self._encourage_buffs[i] = kept

    def _eligible_mask_for_turn(self) -> np.ndarray:
        """
        An agent is eligible to start speaking if their energy is at least their min_energy_to_speak.
        Returns a bool array of shape (3,).
        """
        mask = []
        for i in range(3):
            min_e = self.agent_params[i]["min_energy_to_speak"]
            mask.append(self.energy[i] >= min_e)
        return np.array(mask, dtype=bool)

    def _softmax(
        self, x: np.ndarray, tau: float = 0.0, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Numerically-stable softmax with optional boolean mask (False => probability 0).
        """
        x = np.asarray(x, dtype=float)
        if mask is None:
            mask = np.ones_like(x, dtype=bool)
        masked = np.where(mask, x, -np.inf)
        # subtract max over eligible for stability
        if not mask.any():
            # no eligible entries; return uniform to avoid NaNs
            return np.ones_like(x) / len(x)
        m = np.max(masked[mask])
        exps = np.exp(
            np.clip((masked - m) / max(1e-8, float(tau)), -100, 100)
        ) * mask.astype(float)
        Z = exps.sum()
        if not np.isfinite(Z) or Z <= 0.0:
            # fallback uniform over eligible
            u = mask.astype(float)
            return u / u.sum()
        return exps / Z

    # def _sample_from_probs(self, probs: np.ndarray) -> int:
    #     """
    #     Draw a single index according to probs (shape (N,)).
    #     """
    #     rng = getattr(self, "rng", None)
    #     if rng is None:
    #         # safety, but you already set rng in __init__
    #         rng = np.random.default_rng(42)
    #         self.rng = rng
    #     return int(rng.choice(len(probs), p=probs))
    #     self.n = n_agents
    #     self.rng = np.random.default_rng(seed)
    def _sample_from_probs(self, probs: np.ndarray, seed: Optional[int] = None) -> int:
        rng = np.random.default_rng(seed)
        agents = np.arange(len(probs))
        return int(rng.choice(agents, p=probs))
    
    def _random_sample(self, energy, agents = [0, 1, 2])-> int :
        #random choise 
        energies = energy
        agents = agents

        # Normalize to probabilities
        total = sum(energies)
        probs = [e / total for e in energies]

        # Sample one agent
        choice = random.choices(agents, weights=probs, k=1)[0]
        return choice
    def _uniform_sample(self, energies, agents = [0, 1, 2], tau=2.0, seed=45)-> int :
        rng = np.random.default_rng(seed)

        energies = np.array(energies, dtype=float)
        total = energies.sum()

        if total <= 0:
            # fallback: uniform choice if no energy
            return rng.choice(agents)

        # normalize to probabilities
        probs = energies / total

        # sample one agent
        choice = rng.choice(agents, p=probs)
        return int(choice)
    def _eligible_mask_for_turn(self) -> np.ndarray:
        """Boolean mask of agents allowed to take the floor this step."""
        energies = np.asarray(self.energy, dtype=float)
        # Example rule: everyone with non-negative energy is eligible
        mask = np.isfinite(energies) & (energies >= 0.0)
        return mask
    def _speak_probs(self) -> np.ndarray:
        """
        Compute next-speaker probabilities using:
        baseline = normalized energy
        momentary = d * exp(-b * skipped_i)
        raw = alpha * baseline + (1-alpha) * momentary
        apply constraints (eligibility, no immediate self-follow)
        normalize to probs
        """
        N = self.num_agents
        energies = np.asarray(self.energy, dtype=float)
        energies = np.clip(np.nan_to_num(energies, nan=0.0), 0.0, None)

        # --- 1) baseline from energies (normalize) ---
        total = energies.sum()
        if total <= 0.0:
            baseline = np.full(N, 1.0 / N)         # fallback: uniform baseline
        else:
            baseline = energies / total

        # --- 2) momentary boost with exponential decay by missed turns ---
        t = self.speak_skipped
        momentary = self.speak_d * np.exp(-self.speak_b * t)   # shape (N,)

        # --- 3) mix baseline and momentary ---
        raw = self.speak_alpha * baseline + (1.0 - self.speak_alpha) * momentary

        # --- 4) constraints: eligibility + forbid immediate self-follow ---
        mask = self._eligible_mask_for_turn()
        if self.current_speaker is not None and self.current_speaker >= 0:
            # The paper treats a whole utterance as a unit; forbid immediate re-pick
            mask[self.current_speaker] = False

        raw = np.where(mask, raw, 0.0)

        # --- 5) normalize to probabilities ---
        Z = raw.sum()
        if not np.isfinite(Z) or Z <= 0.0:
            # fallback: uniform over eligible agents
            u = mask.astype(float)
            u_sum = u.sum()
            if u_sum <= 0:
                return np.full(N, 1.0 / N)
            return u / u_sum
        return raw / Z

    # def _speak_probs(self) -> np.ndarray:
    #     """
    #     Compute next-speaker probabilities with energy as baseline propensity.
    #     """
    #     energies = np.array(self.energy, dtype=float)
    #     energies = np.clip(energies, 0.0, None)
    #     if energies.sum() == 0:
    #         energies[:] = 1.0   # fallback if everyone has 0 energy

    #     # 1) baseline from energies (normalized)
    #     pi = energies / energies.sum()

    #     # 2) momentary boost with exponential decay
    #     t = self.speak_skipped
    #     momentary = self.speak_d * np.exp(-self.speak_b * t)

    #     # 3) mix baseline and momentary
    #     raw = self.speak_alpha * pi + (1.0 - self.speak_alpha) * momentary

    #     # 4) apply constraints
    #     mask = self._eligible_mask_for_turn()
    #     if self.current_speaker != -1:
    #         mask[self.current_speaker] = False
    #     raw = np.where(mask, raw, 0.0)

    #     # 5) normalize
    #     Z = raw.sum()
    #     if Z <= 0:
    #         u = mask.astype(float)
    #         return u / max(1.0, u.sum())
    #     return raw / Z
