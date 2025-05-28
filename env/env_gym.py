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


# ------------------------------------------------------------------------
# 1.  ENVIRONMENT
# ------------------------------------------------------------------------

ACTIONS = {
    0: "wait",
    1: "stop",
    2: "stare_at 0",
    3: "stare_at 1",
    4: "stare_at 2",
    5: "encourage 0",
    6: "encourage 1",
    7: "encourage 2",
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
                 energy_imbalance: float = 0.0):  # 0.0 to 1.0, controls initial energy imbalance
        super().__init__()
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.imbalance_factor = imbalance_factor
        self.energy_imbalance = energy_imbalance

        # Action space: 8 discrete Guest actions
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation space: [energy, speaking_time, total_phonemes] * 3
        high = np.array([1.0, 1.0, np.inf] * 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)

        # Agent-specific parameters
        self.agent_params = {
            0: {  # Reserved agent
                'min_energy_to_speak': 0.4,
                'energy_decay': 0.08,
                'energy_gain': 0.04,
                'max_speaking_time': 4,
                'phonemes_per_step': 2
            },
            1: {  # Balanced agent
                'min_energy_to_speak': 0.3,
                'energy_decay': 0.1,
                'energy_gain': 0.05,
                'max_speaking_time': 5,
                'phonemes_per_step': 1
            },
            2: {  # Energetic agent
                'min_energy_to_speak': 0.25,
                'energy_decay': 0.12,
                'energy_gain': 0.06,
                'max_speaking_time': 6,
                'phonemes_per_step': 1
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
        self.phoneme_history = []

    def _get_obs(self) -> np.ndarray:
        obs = []
        for i in range(3):
            obs.extend([
                self.energy[i],
                self.speaking_time[i] / self.agent_params[i]['max_speaking_time'],
                float(self.phonemes[i]),
            ])
        return np.asarray(obs, dtype=np.float32)

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

    def step(self, action: int):
        self.step_counter += 1
        self.action_stats[action] += 1

        # Process Guest action with imbalance factor
        if action == 1 and self.current_speaker != -1:  # stop
            self.energy[self.current_speaker] = 0.0
            self.current_speaker = -1
        elif 2 <= action <= 4:  # stare_at
            target = action - 2
            effect = 0.2 * (1 - self.imbalance_factor)
            self.energy[target] = min(1.0, self.energy[target] + effect)
        elif 5 <= action <= 7:  # encourage
            target = action - 5
            if self.current_speaker == -1:
                effect = 0.3 * (1 - self.imbalance_factor)
                self.energy[target] = min(1.0, self.energy[target] + effect)

        # Agent-specific energy dynamics
        for i in range(3):
            params = self.agent_params[i]
            if i != self.current_speaker:
                gain = params['energy_gain'] * (1 + self.imbalance_factor * (i - 1))
                self.energy[i] = min(1.0, self.energy[i] + gain)
            else:
                self.energy[i] = max(0.0, self.energy[i] - params['energy_decay'])

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
        self.phoneme_history.append(self.phonemes.copy())

        return obs, reward, terminated, truncated, info
