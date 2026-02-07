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
import os
from datetime import datetime
import math

logger.remove()
logger.add(
    sys.stdout, colorize=True, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <blue>{level}</blue> | <level>{message}</level>"
)
date_str = datetime.now().strftime("%Y_%m_%d")
logger.add(f"{os.getcwd()}/training_{date_str}.log", format="{time} | {level} | {message}", level="TRACE")
# logger.add(
#     sys.stdout,
#     colorize=True,
#     format="<green>{time}</green> <level>{message}</level>",
#     level="INFO",
# )

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
        seed = 42,
        imbalance_factor: float = 0.0,  # 0.0 to 1.0, controls natural imbalance
        energy_imbalance: float = 0.0,
        reward_shaping=False,
        logfile=False,
        encourage_base_effect: float = 0.9,
        encourage_duration_steps: int = 10,
        encourage_stack: bool = True,
        env_effect: bool = True,
        env_effect_encourage_step: int = 10,
        efficiency: bool = False,
        randomize_agent: bool = False 
    ):  # 0.0 to 1.0, controls initial energy imbalance
        super().__init__()
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.imbalance_factor = imbalance_factor  # It gives you a single knob (imbalance_factor) to favor “higher-index” agents (2) over “lower-index” ones (0), with agent 1 in the middle.
        self.energy_imbalance = energy_imbalance
        self.reward_shaping = reward_shaping
        self.efficiency = efficiency
        self.randomize_agent = randomize_agent

        # Action space: 7 discrete Guest actions
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation space: [energy, speaking_time, total_phonemes] * 3
        # high = np.array([1.0, 1.0, np.inf] * 3, dtype=np.float32)
        # self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # Agent-specific parameters
        self.agent_params = {
            0: {  # Reserved agent
                "min_energy_to_speak": 0.20,
                "energy_gain": 0.002,
                "energy_decay": 0.08,
                "max_speaking_time": 6,
                "phonemes_per_step": 4,
            },
            1: {  # Balanced agent
                "min_energy_to_speak": 0.55,
                "energy_gain": 0.010,
                "energy_decay": 0.05,
                "max_speaking_time": 8,
                "phonemes_per_step": 5,
            },
            2: {  # Energetic agent
                "min_energy_to_speak": 0.85,
                "energy_gain": 0.028,
                "energy_decay": 0.03,
                "max_speaking_time": 12,
                "phonemes_per_step": 6,
            },
        }

        # temporary buff
        self.encourage_base_effect = float(encourage_base_effect)
        self.encourage_duration_steps = int(encourage_duration_steps)
        self.encourage_stack = bool(encourage_stack)

        # active buffs per speaker: list[dict(amount, remaining)]
        self._encourage_buffs = [[] for _ in range(len(self.agent_params))]

        # State variables
        self.energy = np.zeros(len(self.agent_params))
        self.speaking_time = np.zeros(len(self.agent_params))
        self.phonemes = np.zeros(len(self.agent_params), dtype=int)
        self.current_speaker = -1
        self.step_counter = 0
        self.action_stats = np.zeros(len(ACTIONS), dtype=int)

        # Imbalance tracking
        self.gini_history = []
        self.env_reward = []
        self.phoneme_history = []

        self.env_effect = env_effect
        self.env_effect_step = 1/max_steps
        self.env_effect_encourage_step = env_effect_encourage_step

        if logfile:
            logger.add("guest_{time:YYYY-MM-DD}.log", enqueue=True)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed)
        self.rng = np.random.default_rng(self.seed)
        if self.randomize_agent:
            self._randomize_params()

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
            low_val = self.rng.uniform(0.0, 0.05)
            rest_vals = self.rng.uniform(0.1, 0.5, size=2)
            self.energy = np.array([low_val, *rest_vals])
            # self.energy = self.rng.uniform(0.01, 0.6, size=3)
            logger.info(f"{self.energy=}")

        logger.info(f"{self.agent_params=}")
        self.speaking_time = np.zeros(len(self.agent_params))
        self.phonemes = np.zeros(len(self.agent_params), dtype=int)
        self.current_speaker = -1
        self.step_counter = 0
        self.action_stats[:] = 0
        self.gini_history = []
        self.phoneme_history = []
        self._encourage_buffs = [[] for _ in range(len(self.agent_params))]

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

    @logger.catch(reraise=True)
    def step(self, action: int):
        self.step_counter += 1
        self.action_stats[action] += 1

        # Process Guest action with imbalance factor
        logger.debug(f"{self.current_speaker=}")
        if 1 <= action <= 3:  # stare_at
            target = action - 1
            effect = 0.01 * (1 - self.imbalance_factor) if not self.env_effect else self.env_effect_step
            logger.debug(f"{effect=}") # TODO remove
            self.energy[target] = min(1.0, self.energy[target] + effect)
            logger.debug(f"{self.energy[target]=}") # TODO remove
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
                # Choose speaker with highest energy
                if not any(self.energy): 
                    logger.info(f"{self.energy=} - step {self.step}")
                choose_speaker = self._random_sample(self.energy)
                self.current_speaker = choose_speaker
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
        efficiency = self._calculate_efficiency()
        gini = self._gini()
        base_reward = 1.0 - gini  # Base reward on equality
        reward = base_reward  # Start with base reward

        if self.efficiency:
            reward = base_reward *efficiency  # Scale by efficiency
        if self.reward_shaping:
            # # Additional reward shaping
            # if self.current_speaker != -1:
            #     # Bonus for new speaker
            #     if self.speaking_time[self.current_speaker] == 1:
            #         reward += 0.2
            #     # Penalty for long speaking turns
            #     if (
            #         self.speaking_time[self.current_speaker]
            #         > self.agent_params[self.current_speaker]["max_speaking_time"] * 0.8
            #     ):
            #         reward -= 0.1
            if action == 0:  # wait
                reward += 0.05  # small penalty for inactivity

        terminated = False
        truncated = self.step_counter >= self.max_steps

        obs = self._get_obs()
        info = dict(
            num_of_step_env=self.step_counter,
            phoneme=self.phonemes.copy(),
            actions_stats=self.action_stats.copy(),
            env_reward=base_reward,  # Base reward without shaping
            total_reward=reward,     # FIXED: Total reward including shaping
            reward_shaping_active=self.reward_shaping,
            current_gini=gini,       # FIXED: Direct gini value
            action_number=int(action),
            gini_history=self.gini_history,
            phoneme_history=self.phoneme_history,
            energy=self.energy.copy(),
        )
        
        # FIXED: Store actual gini coefficient, not inverted
        self.gini_history.append(gini)
        self.env_reward.append(base_reward)
        self.phoneme_history.append(self.phonemes.copy())

        # Log after each step
        logger.trace(f"Step={self.step_counter} | Reward={reward:.4f} | Phonemes={self.phonemes.tolist()}")

        
        self._tick_encourage_buffs()
        return obs, reward, terminated, truncated, info

    # ---- NEW helper methods somewhere in the class ----

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
        diffs = np.abs(x[:, None] - x[None, :]).sum() # n*n
        n = len(x)
        return float(diffs / (2 * n * total))

    # def _apply_encourage(self, target: int) -> None:
    #     """Apply a temporary encourage buff to `target`."""

    #     if not self.env_effect:
    #         effect = self.encourage_base_effect * (1 - self.imbalance_factor)
    #     else:
    #         effect = self.env_effect_step * self.env_effect_encourage_step

    #     headroom = 1.0 - float(self.energy[target])
    #     delta = max(0.0, min(effect, headroom))
    #     if delta <= 0.0:
    #         return

    #     # apply now
    #     # logger.info(f"{target=}")
    #     # logger.info(f"{delta=}")
    #     self.energy[target] = min(0.9, float(self.energy[target]) + delta)

    #     # record the buff so we can roll it back after N steps
    #     buff = {"amount": float(delta), "remaining": int(self.encourage_duration_steps)}
    #     # logger.info(f"{buff=}")
    #     if self.encourage_stack:
    #         self._encourage_buffs[target].append(buff)
    #     else:
    #         # replace any existing buff: first remove them, then set a single new one
    #         for b in self._encourage_buffs[target]:
    #             self.energy[target] = max(
    #                 0.0, min(0.9, float(self.energy[target]) - b["amount"])
    #             )
    #         self._encourage_buffs[target] = [buff]

    # def _tick_encourage_buffs(self) -> None:
    #     """Decrement timers and remove expired buff amounts from energy."""
    #     for i in range(len(self.agent_params)):
    #         kept = []
    #         for b in self._encourage_buffs[i]:
    #             b["remaining"] -= 1
    #             if b["remaining"] <= 0:
    #                 # expire: remove exactly what we added
    #                 self.energy[i] = max(
    #                     0.0, min(1.0, float(self.energy[i]) - b["amount"])
    #                 )
    #             else:
    #                 kept.append(b)
    #         self._encourage_buffs[i] = kept
    def _apply_encourage(self, target: int) -> None:
        """Apply a temporary encourage buff to `target`."""

        # 1. Compute base effect
        if not self.env_effect:
            effect = self.encourage_base_effect * (1 - self.imbalance_factor)
        else:
            effect = self.env_effect_step * self.env_effect_encourage_step

        # 2. Respect max energy cap
        max_energy = 0.9
        current = float(self.energy[target])
        headroom = max_energy - current
        delta = max(0.0, min(effect, headroom))

        if delta <= 0.0:
            return

        # 3. Apply and compute the real applied delta (for perfect rollback)
        new_energy = current + delta
        new_energy = min(max_energy, new_energy)
        delta_applied = new_energy - current

        if delta_applied <= 0.0:
            return

        self.energy[target] = new_energy

        # 4. Record buff
        buff = {"amount": float(delta_applied), "remaining": int(self.encourage_duration_steps)}

        if self.encourage_stack:
            self._encourage_buffs[target].append(buff)
        else:
            # Remove any existing buffs first
            for b in self._encourage_buffs[target]:
                self.energy[target] = max(
                    0.0, min(max_energy, float(self.energy[target]) - b["amount"])
                )
            self._encourage_buffs[target] = [buff]


    def _tick_encourage_buffs(self) -> None:
        """Decrement timers and remove expired buff amounts from energy."""
        max_energy = 0.9
        for i in range(len(self.agent_params)):
            kept = []
            for b in self._encourage_buffs[i]:
                b["remaining"] -= 1
                if b["remaining"] <= 0:
                    # expire: remove exactly what we added
                    self.energy[i] = max(
                        0.0, min(max_energy, float(self.energy[i]) - b["amount"])
                    )
                else:
                    kept.append(b)
            self._encourage_buffs[i] = kept

    
    def _random_sample(self, energy, agents = [0, 1, 2])-> int :
        #random choise 
        energies = [float(energy[i]) for i in agents]

        # Compute total on finite values only
        finite_energies = [e for e in energies if math.isfinite(e)]
        total = sum(finite_energies)

        if not math.isfinite(total) or total <= 0:
            return random.choice(agents)
        
        # Normalize to probabilities
        # total = sum(energies)
        probs = [e / total for e in energies]

        # Sample one agent
        choice = random.choices(agents, weights=probs, k=1)[0]
        return choice
    
    def _calculate_efficiency(self) -> float:
        total_phonemes = np.sum(self.phonemes)
        
        # Early episodes: no efficiency penalty
        if total_phonemes == 0:
            return 1.0
        
        # Calculate ideal phonemes per agent (equal distribution)
        ideal_per_agent = total_phonemes / 3
        
        # Calculate how close each agent is to ideal
        deviations = np.abs(self.phonemes - ideal_per_agent)
        mean_deviation = np.mean(deviations)
        
        # Normalize deviation (max deviation is when one agent has all phonemes)
        max_possible_deviation = (2 * total_phonemes) / 3
        normalized_deviation = mean_deviation / max_possible_deviation if max_possible_deviation > 0 else 0.0
        
        # Convert to efficiency score (1.0 - deviation)
        efficiency = 1.0 - normalized_deviation
        
        return float(np.clip(efficiency, 0.0, 1.0))
    
    def _randomize_agent(self):
        """
        Creates 3 distinct personalities and randomly assigns them 
        to Agent 0, 1, and 2.
        """
        
        # 1. Define the Three Archetypes (The "Souls")
        # We still use ranges so they are distinct but slightly variable
        
        # Soul A: The Reserved One
        reserved_soul = {
            # "type": "RESERVED",  # Optional: Helpful for debugging/logging
            "min_energy_to_speak": 0.20,
            "energy_gain": 0.0018,
            "energy_decay": 0.09,
            "max_speaking_time": 6,
            "phonemes_per_step": 4,
        }

        # Soul B: The Balanced One
        balanced_soul = {
            # "type": "BALANCED",
            "min_energy_to_speak": 0.55,
            "energy_gain": 0.010,
            "energy_decay": 0.05,
            "max_speaking_time": 8,
            "phonemes_per_step": 5,
        }

        # Soul C: The Energetic One
        energetic_soul = {
            # "type": "ENERGETIC",
            "min_energy_to_speak": 0.85,
            "energy_gain": 0.030,
            "energy_decay": 0.025,
            "max_speaking_time": 12,
            "phonemes_per_step": 6,
        }

        # 2. Put souls in a list
        available_souls = [reserved_soul, balanced_soul, energetic_soul]

        # 3. SHUFFLE THEM (The Switch)
        # This is where Agent 0 might get the 'Energetic' soul
        np_random = np.random.default_rng(None)
        np_random.shuffle(available_souls)

        # 4. Assign to bodies (Agent 0, 1, 2)
        self.agent_params = {
            0: available_souls[0],
            1: available_souls[1],
            2: available_souls[2]
        }
        logger.info(f"in function - {self.agent_params=}")
        
        # Optional: Log who is who for this episode
        # logger.info(f"Episode Setup: Agent0={self.agent_params[0]['type']}, Agent1={self.agent_params[1]['type']}, Agent2={self.agent_params[2]['type']}")
    def _randomize_params(self):
        """
        Creates 3 distinct personalities and randomly assigns them 
        to Agent 0, 1, and 2.
        """
        
        # 1. Define the Three Archetypes (The "Souls")
        # We still use ranges so they are distinct but slightly variable
        
        # Soul A: The Reserved One
        np_random = np.random.default_rng(None)
        reserved_soul = {
            "type": "RESERVED",  # Optional: Helpful for debugging/logging
            "min_energy_to_speak": np_random.uniform(0.0, 1.0),
            "energy_gain":         np_random.uniform(0.0, 0.5),
            "energy_decay":        np_random.uniform(0.0, 0.5),
            "max_speaking_time":   np_random.integers(3, 10),
            "phonemes_per_step":   np_random.integers(3, 10),
        }

        # Soul B: The Balanced One
        balanced_soul = {
            "type": "BALANCED",
            "min_energy_to_speak": np_random.uniform(0.0, 1.0),
            "energy_gain":         np_random.uniform(0.0, 0.5),
            "energy_decay":        np_random.uniform(0.0, 0.5),
            "max_speaking_time":   np_random.integers(3, 10),
            "phonemes_per_step":   np_random.integers(3, 10),
        }

        # Soul C: The Energetic One
        energetic_soul = {
            "type": "ENERGETIC",
            "min_energy_to_speak": np_random.uniform(0.0, 1.0),
            "energy_gain":         np_random.uniform(0.0, 0.5),
            "energy_decay":        np_random.uniform(0.0, 0.5),
            "max_speaking_time":   np_random.integers(3, 10),
            "phonemes_per_step":   np_random.integers(3, 10),
        }

        # 2. Put souls in a list
        available_souls = [reserved_soul, balanced_soul, energetic_soul]

        # 3. SHUFFLE THEM (The Switch)
        # This is where Agent 0 might get the 'Energetic' soul
        np_random.shuffle(available_souls)

        # 4. Assign to bodies (Agent 0, 1, 2)
        self.agent_params = {
            0: available_souls[0],
            1: available_souls[1],
            2: available_souls[2]
        }
        logger.info(f"{self.agent_params=}")
        
        # Optional: Log who is who for this episode
        # logger.info(f"Episode Setup: Agent0={self.agent_params[0]['type']}, Agent1={self.agent_params[1]['type']}, Agent2={self.agent_params[2]['type']}")