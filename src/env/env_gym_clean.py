"""
Clean GuestEnv - Conversation balancing environment
Features:
- 3 agents with identical parameters
- Inequality driven by initial energy (compound growth)
- encourage: temporary energy buff (expires after N steps)
- stare_at: small permanent energy increase
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>"
)

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
    Conversation environment with 3 identical agents.
    Inequality emerges from initial energy differences via compound growth.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_steps: int = 600,
        seed: int = 42,
        initial_energy: np.ndarray = None,  # [agent0, agent1, agent2]

        # Agent parameters (same for all)
        min_energy_to_speak: float = 0.55,
        energy_gain_rate: float = 0.012,      # Base rate for energy recovery
        energy_decay: float = 0.05,
        max_speaking_time: int = 8,
        phonemes_per_step: int = 5,

        # Action effects
        stare_boost: float = 0.015,           # Small permanent boost
        encourage_boost: float = 0.12,        # Temporary boost amount
        encourage_duration: int = 10,         # Steps before boost expires

        # Dynamics mode
        use_compound_growth: bool = True,     # Slightly favor higher energy (realistic)

        # Reward settings
        reward_shaping: bool = False,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Store initial energy config
        self.initial_energy = initial_energy

        # Agent parameters (identical for all 3)
        self.agent_params = {
            "min_energy_to_speak": min_energy_to_speak,
            "energy_gain_rate": energy_gain_rate,
            "energy_decay": energy_decay,
            "max_speaking_time": max_speaking_time,
            "phonemes_per_step": phonemes_per_step,
        }

        # Action effects
        self.stare_boost = stare_boost
        self.encourage_boost = encourage_boost
        self.encourage_duration = encourage_duration

        # Dynamics
        self.use_compound_growth = use_compound_growth

        # Reward
        self.reward_shaping = reward_shaping

        # Gym spaces
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # State variables
        self.energy = np.zeros(3)
        self.speaking_time = np.zeros(3)
        self.phonemes = np.zeros(3, dtype=int)
        self.current_speaker = -1
        self.step_counter = 0
        self.action_stats = np.zeros(len(ACTIONS), dtype=int)

        # Encourage buffs: list of {amount, remaining_steps} per agent
        self.encourage_buffs = [[] for _ in range(3)]

        # Tracking
        self.gini_history = []
        self.phoneme_history = []

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=self.seed)
        self.rng = np.random.default_rng(self.seed)

        # Initialize energy
        if self.initial_energy is not None:
            self.energy = np.array(self.initial_energy, dtype=float)
        else:
            # Default: random energy with one agent starting low
            self.energy = self.rng.uniform(0.1, 0.6, size=3)

        logger.info(f"Initial energy: {self.energy}")
        logger.info(f"Agent params: {self.agent_params}")

        # Reset state
        self.speaking_time = np.zeros(3)
        self.phonemes = np.zeros(3, dtype=int)
        self.current_speaker = -1
        self.step_counter = 0
        self.action_stats[:] = 0
        self.encourage_buffs = [[] for _ in range(3)]
        self.gini_history = []
        self.phoneme_history = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        """Execute one environment step."""
        self.step_counter += 1
        self.action_stats[action] += 1

        # 1. GUEST ACTION
        if 1 <= action <= 3:  # stare_at
            target = action - 1
            self.energy[target] = min(1.0, self.energy[target] + self.stare_boost)
            logger.debug(f"Stare at agent {target}: energy now {self.energy[target]:.3f}")

        elif 4 <= action <= 6:  # encourage
            target = action - 4
            self._apply_encourage(target)

        # 2. ENERGY DYNAMICS (Realistic conversation dynamics)
        params = self.agent_params
        for i in range(3):
            if i != self.current_speaker:
                # Energy recovery while not speaking
                if self.use_compound_growth:
                    # Compound growth: confidence builds on confidence
                    # Higher energy → faster recovery (realistic: confident people speak up more)
                    # Using direct proportional: gain is proportional to current energy
                    gain = params["energy_gain_rate"] * self.energy[i]
                else:
                    # Linear growth (everyone recovers at same rate)
                    gain = params["energy_gain_rate"] * 0.5  # Fixed rate

                self.energy[i] = min(1.0, self.energy[i] + gain)
            else:
                # Speaker loses energy while talking
                self.energy[i] = max(0.0, self.energy[i] - params["energy_decay"])

        # 3. SPEAKING DYNAMICS
        if self.current_speaker == -1:
            # Find who can speak (above threshold)
            candidates = [i for i in range(3)
                         if self.energy[i] >= params["min_energy_to_speak"]]

            if len(candidates) > 0:
                # Choose highest energy agent
                self.current_speaker = candidates[np.argmax(self.energy[candidates])]
                self.speaking_time[self.current_speaker] = 0
                logger.debug(f"Agent {self.current_speaker} starts speaking")
        else:
            # Current speaker continues or yields
            self.speaking_time[self.current_speaker] += 1

            if (self.speaking_time[self.current_speaker] >= params["max_speaking_time"] or
                self.energy[self.current_speaker] < params["min_energy_to_speak"]):
                logger.debug(f"Agent {self.current_speaker} stops speaking")
                self.current_speaker = -1

        # 4. UPDATE PHONEMES
        if self.current_speaker != -1:
            self.phonemes[self.current_speaker] += params["phonemes_per_step"]

        # 5. TICK ENCOURAGE BUFFS (expire temporary boosts)
        self._tick_encourage_buffs()

        # 6. CALCULATE REWARD
        gini = self._calculate_gini()
        base_reward = 1.0 - gini  # Higher reward for equality
        reward = base_reward

        if self.reward_shaping and action == 0:  # Small penalty for waiting
            reward -= 0.05

        # 7. TERMINATION
        terminated = False
        truncated = self.step_counter >= self.max_steps

        # 8. TRACKING
        self.gini_history.append(gini)
        self.phoneme_history.append(self.phonemes.copy())

        obs = self._get_obs()
        info = self._get_info(action, reward, gini)

        return obs, reward, terminated, truncated, info

    def _apply_encourage(self, target: int):
        """Apply temporary energy boost to target agent."""
        current_energy = float(self.energy[target])

        # Calculate boost (capped at 0.9 max energy)
        max_energy = 0.9
        headroom = max_energy - current_energy
        actual_boost = min(self.encourage_boost, headroom)

        if actual_boost <= 0:
            return

        # Apply boost
        self.energy[target] = current_energy + actual_boost

        # Record buff for later removal
        buff = {
            "amount": float(actual_boost),
            "remaining": int(self.encourage_duration)
        }
        self.encourage_buffs[target].append(buff)

        logger.debug(f"Encourage agent {target}: +{actual_boost:.3f} energy for {self.encourage_duration} steps")

    def _tick_encourage_buffs(self):
        """Decrement buff timers and remove expired boosts."""
        for i in range(3):
            kept_buffs = []
            for buff in self.encourage_buffs[i]:
                buff["remaining"] -= 1

                if buff["remaining"] <= 0:
                    # Buff expired - remove the energy boost
                    self.energy[i] = max(0.0, self.energy[i] - buff["amount"])
                    logger.debug(f"Encourage expired for agent {i}: -{buff['amount']:.3f}")
                else:
                    kept_buffs.append(buff)

            self.encourage_buffs[i] = kept_buffs

    def _calculate_gini(self) -> float:
        """Calculate Gini coefficient for phoneme distribution."""
        total = np.sum(self.phonemes)
        if total == 0:
            return 0.0

        x = self.phonemes.astype(float)
        diffs = np.abs(x[:, None] - x[None, :]).sum()
        n = len(x)
        return float(diffs / (2 * n * total))

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        obs = []

        # 1. Energy levels [3]
        obs.extend(self.energy)

        # 2. Speaking time ratios [3]
        for i in range(3):
            ratio = self.speaking_time[i] / self.agent_params["max_speaking_time"]
            obs.append(min(1.0, ratio))

        # 3. Phoneme distribution [3]
        total_phonemes = np.sum(self.phonemes)
        if total_phonemes > 0:
            phoneme_dist = self.phonemes / total_phonemes
        else:
            phoneme_dist = np.ones(3) / 3
        obs.extend(phoneme_dist)

        # 4. Current speaker one-hot [4] - [nobody, agent0, agent1, agent2]
        speaker_encoding = np.zeros(4)
        if self.current_speaker == -1:
            speaker_encoding[0] = 1.0
        else:
            speaker_encoding[self.current_speaker + 1] = 1.0
        obs.extend(speaker_encoding)

        # 5. Gini coefficient [1]
        obs.append(self._calculate_gini())

        # 6. Phoneme stats [2]
        if total_phonemes > 0:
            mean_phonemes = total_phonemes / 3
            phoneme_std = np.std(self.phonemes) / mean_phonemes
            phoneme_range = (np.max(self.phonemes) - np.min(self.phonemes)) / mean_phonemes
        else:
            phoneme_std = 0.0
            phoneme_range = 0.0
        obs.extend([phoneme_std, phoneme_range])

        # 7. Progress [1]
        progress = self.step_counter / self.max_steps
        obs.append(progress)

        return np.asarray(obs, dtype=np.float32)

    def _get_info(self, action=-1, reward=0.0, gini=0.0):
        """Build info dictionary."""
        return {
            "step": self.step_counter,
            "phonemes": self.phonemes.copy(),
            "energy": self.energy.copy(),
            "action": int(action),
            "action_stats": self.action_stats.copy(),
            "reward": float(reward),
            "gini": float(gini),
            "gini_history": self.gini_history.copy(),
            "phoneme_history": [p.copy() for p in self.phoneme_history],
        }
