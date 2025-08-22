# guest_rl_train.py
# -----------------------------------------------------------
# pip install gymnasium stable-baselines3 torch tensorboard
# -----------------------------------------------------------

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO",format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", level="INFO")

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

    def __init__(self, *, max_steps: int = 600, seed: int = 42, 
                 imbalance_factor: float = 0.0,  # 0.0 to 1.0, controls natural imbalance
                 energy_imbalance: float = 0.0,
                 reward_shaping = True,
                 logfile = False,
                 encourage_base_effect: float = 0.3,
                 encourage_duration_steps: int = 10,
                 encourage_stack: bool = True):  # 0.0 to 1.0, controls initial energy imbalance
        super().__init__()
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.imbalance_factor = imbalance_factor #It gives you a single knob (imbalance_factor) to favor “higher-index” agents (2) over “lower-index” ones (0), with agent 1 in the middle.
        self.energy_imbalance = energy_imbalance
        self.reward_shaping = reward_shaping

        # Action space: 8 discrete Guest actions
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation space: [energy, speaking_time, total_phonemes] * 3
        # high = np.array([1.0, 1.0, np.inf] * 3, dtype=np.float32)
        # self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)

        # temporary buff
        self.encourage_base_effect = float(encourage_base_effect)
        self.encourage_duration_steps = int(encourage_duration_steps)
        self.encourage_stack = bool(encourage_stack)

        # active buffs per speaker: list[dict(amount, remaining)]
        self._encourage_buffs = [[] for _ in range(3)]

        # Agent-specific parameters
        self.agent_params = {
            0: {  # Reserved agent
                'min_energy_to_speak': 0.5,
                'energy_decay': 0.05,
                'energy_gain': 0.04,
                'max_speaking_time': 5,
                'phonemes_per_step': 3
            },
            1: {  # Balanced agent
                'min_energy_to_speak': 0.5,
                'energy_decay': 0.5,
                'energy_gain': 0.05,
                'max_speaking_time': 5,
                'phonemes_per_step': 3
            },
            2: {  # Energetic agent
                'min_energy_to_speak': 0.9,
                'energy_decay': 0.2,
                'energy_gain': 0.06,
                'max_speaking_time': 5,
                'phonemes_per_step': 3
            }
        }

        # State variables
        self.energy = np.zeros(3)
        self.speaking_time = np.zeros(3)
        self.phonemes = np.zeros(3, dtype=int)
        self.current_speaker = -1
        self.step_counter = 0
        self.action_stats = np.zeros(len(ACTIONS), dtype=int)
        
        # Imbalance tracking
        self.gini_history = []
        self.env_reward = []
        self.phoneme_history = []
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
            ratio = self.speaking_time[i] / self.agent_params[i]['max_speaking_time']
            speaking_ratios.append(min(1.0, ratio))
        obs.extend(speaking_ratios)  # [3 values: 0-1]
        
        # 3. PHONEME DISTRIBUTION (softmax normalized)
        total_phonemes = np.sum(self.phonemes)
        if total_phonemes > 0:
            phoneme_distribution = self.phonemes / total_phonemes  # Relative proportions
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
            phoneme_range = (np.max(self.phonemes) - np.min(self.phonemes)) / (total_phonemes / 3)
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
            self.energy = np.array([
                base_energy * (1 - self.energy_imbalance),
                base_energy,
                base_energy * (1 + self.energy_imbalance)
            ])
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
            phoneme_history=self.phoneme_history
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
            effect = 0.2 * (1 - self.imbalance_factor)
            self.energy[target] = min(1.0, self.energy[target] + effect)
        elif 4 <= action <= 6:  # encourage
            target = action - 4
            self._apply_encourage(target)
            # if self.current_speaker == -1:
            #     effect = 0.3 * (1 - self.imbalance_factor)
            #     self.energy[target] = min(1.0, self.energy[target] + effect)

        # Agent-specific energy dynamics
        for i in range(3):
            params = self.agent_params[i]
            if i != self.current_speaker:
                gain = params['energy_gain'] * (1 + self.imbalance_factor * (i - 1))
                self.energy[i] = min(1.0, self.energy[i] + gain)
                logger.debug(f"{self.energy[i]}")
            else:
                self.energy[i] = max(0.0, self.energy[i] - params['energy_decay'])
                logger.debug(f"{self.energy[i]}")

        # Speaking dynamics with agent-specific parameters
        if self.current_speaker == -1:
            # Find potential speakers based on their individual thresholds
            candidates = []
            for i in range(3):
                if self.energy[i] >= self.agent_params[i]['min_energy_to_speak']:
                    candidates.append(i)
            
            if len(candidates) > 0:
                # Choose speaker with highest energy
                self.current_speaker = candidates[np.argmax(self.energy[candidates])]
                logger.debug(f"{self.current_speaker=}")
                self.speaking_time[self.current_speaker] = 0
        else:
            # Current speaker continues or yields based on their parameters
            params = self.agent_params[self.current_speaker]
            self.speaking_time[self.current_speaker] += 1
            if (self.speaking_time[self.current_speaker] >= params['max_speaking_time'] or 
                self.energy[self.current_speaker] < params['min_energy_to_speak']):
                self.current_speaker = -1

        # Update phonemes with agent-specific rates
        if self.current_speaker != -1:
            self.phonemes[self.current_speaker] += self.agent_params[self.current_speaker]['phonemes_per_step']

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
                if self.speaking_time[self.current_speaker] > self.agent_params[self.current_speaker]['max_speaking_time'] * 0.8:
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
            phoneme_history=self.phoneme_history
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
        if self.current_speaker != -1:
            return  # your existing logic: only when nobody is speaking
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
                self.energy[target] = max(0.0, min(1.0, float(self.energy[target]) - b["amount"]))
            self._encourage_buffs[target] = [buff]

    def _tick_encourage_buffs(self) -> None:
        """Decrement timers and remove expired buff amounts from energy."""
        for i in range(3):
            kept = []
            for b in self._encourage_buffs[i]:
                b["remaining"] -= 1
                if b["remaining"] <= 0:
                    # expire: remove exactly what we added
                    self.energy[i] = max(0.0, min(1.0, float(self.energy[i]) - b["amount"]))
                else:
                    kept.append(b)
            self._encourage_buffs[i] = kept
