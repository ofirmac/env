"""
Professional RL Training Script for GuestEnv
============================================
Clean, modular training script with best practices for RL training.

Features:
- Configuration management
- Multiple training modes (dev/prod/fast)
- Command line arguments
- Resume training capability
- Comprehensive logging
- Model evaluation
- Hyperparameter tracking
"""

import os
import sys
import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env.env_gym import GuestEnv
from src.callback.guest_callback_per_episode_multi_env import CallbackPerEpisode


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EnvConfig:
    """Environment configuration"""
    max_steps: int = 500
    reward_shaping: bool = False
    env_effect: bool = False
    efficiency: bool = False
    randomize_agent: bool = False
    seed: int = 42

    # Agent parameters
    agent_0_min_energy: float = 0.20
    agent_0_energy_gain: float = 0.002
    agent_0_energy_decay: float = 0.08
    agent_0_max_speaking: int = 6
    agent_0_phonemes: int = 4

    agent_1_min_energy: float = 0.55
    agent_1_energy_gain: float = 0.010
    agent_1_energy_decay: float = 0.05
    agent_1_max_speaking: int = 8
    agent_1_phonemes: int = 5

    agent_2_min_energy: float = 0.85
    agent_2_energy_gain: float = 0.028
    agent_2_energy_decay: float = 0.03
    agent_2_max_speaking: int = 12
    agent_2_phonemes: int = 6


@dataclass
class PPOConfig:
    """PPO algorithm configuration"""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range_vf: Optional[float] = None
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "auto"


@dataclass
class TrainingConfig:
    """Training configuration"""
    total_episodes: int = 1000
    num_parallel_envs: int = 1
    num_envs_to_track: int = 1  # How many envs to save to pkl (1=only first env, 4=all envs)
    eval_freq: int = 10000
    save_freq: int = 50000
    eval_episodes: int = 10
    test_after_training: bool = True
    test_episodes: int = 50
    use_tensorboard: bool = True
    save_pkl: bool = True  # Whether to save callback data to pkl
    verbose: int = 1


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_config_preset(preset: str = "default") -> tuple[EnvConfig, PPOConfig, TrainingConfig]:
    """Get predefined configuration presets"""

    if preset == "dev":
        # Fast iterations for development
        env_cfg = EnvConfig(max_steps=200)
        ppo_cfg = PPOConfig(
            n_steps=512,
            batch_size=128,
            n_epochs=3,
        )
        train_cfg = TrainingConfig(
            total_episodes=100,
            eval_freq=5000,
            save_freq=10000,
        )

    elif preset == "fast":
        # Quick training, less thorough
        env_cfg = EnvConfig(max_steps=500)
        ppo_cfg = PPOConfig(
            n_steps=1024,
            batch_size=128,
            n_epochs=3,
        )
        train_cfg = TrainingConfig(
            total_episodes=500,
            num_parallel_envs=4,
            num_envs_to_track=1,  # Track only 1 env to save memory
        )

    elif preset == "production":
        # Thorough training for best results
        env_cfg = EnvConfig(max_steps=600)
        ppo_cfg = PPOConfig(
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
        )
        train_cfg = TrainingConfig(
            total_episodes=5000,
            num_parallel_envs=4,
            num_envs_to_track=4,  # Track all envs for complete data
            eval_freq=20000,
        )

    else:  # default
        env_cfg = EnvConfig()
        ppo_cfg = PPOConfig()
        train_cfg = TrainingConfig()

    return env_cfg, ppo_cfg, train_cfg


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def make_env(env_config: EnvConfig, rank: int = 0, logfile: bool = False):
    """Create a single environment instance"""
    def _init():
        env = GuestEnv(
            max_steps=env_config.max_steps,
            reward_shaping=env_config.reward_shaping,
            env_effect=env_config.env_effect,
            efficiency=env_config.efficiency,
            randomize_agent=env_config.randomize_agent,
            seed=env_config.seed + rank,
            logfile=logfile,
        )

        # Configure agent parameters
        env.agent_params[0].update({
            "min_energy_to_speak": env_config.agent_0_min_energy,
            "energy_gain": env_config.agent_0_energy_gain,
            "energy_decay": env_config.agent_0_energy_decay,
            "max_speaking_time": env_config.agent_0_max_speaking,
            "phonemes_per_step": env_config.agent_0_phonemes,
        })

        env.agent_params[1].update({
            "min_energy_to_speak": env_config.agent_1_min_energy,
            "energy_gain": env_config.agent_1_energy_gain,
            "energy_decay": env_config.agent_1_energy_decay,
            "max_speaking_time": env_config.agent_1_max_speaking,
            "phonemes_per_step": env_config.agent_1_phonemes,
        })

        env.agent_params[2].update({
            "min_energy_to_speak": env_config.agent_2_min_energy,
            "energy_gain": env_config.agent_2_energy_gain,
            "energy_decay": env_config.agent_2_energy_decay,
            "max_speaking_time": env_config.agent_2_max_speaking,
            "phonemes_per_step": env_config.agent_2_phonemes,
        })

        # Wrap with Monitor for episode statistics
        env = Monitor(env)
        return env

    set_random_seed(env_config.seed + rank)
    return _init


def create_vectorized_env(env_config: EnvConfig, num_envs: int = 1, use_subprocess: bool = True):
    """Create vectorized environment for parallel training"""
    if num_envs > 1 and use_subprocess:
        print(f"Creating {num_envs} parallel environments (SubprocVecEnv)...")
        env = SubprocVecEnv([make_env(env_config, i) for i in range(num_envs)])
    else:
        if num_envs > 1:
            print(f"Creating {num_envs} parallel environments (DummyVecEnv)...")
            env = DummyVecEnv([make_env(env_config, i) for i in range(num_envs)])
        else:
            print("Creating single environment...")
            env = DummyVecEnv([make_env(env_config, 0)])

    return env


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Professional trainer class with all training logic"""

    def __init__(
        self,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        train_config: TrainingConfig,
        results_dir: Path,
        resume_from: Optional[str] = None,
    ):
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.train_config = train_config
        self.results_dir = Path(results_dir)
        self.resume_from = resume_from

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.results_dir / "tensorboard"
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.logs_dir = self.results_dir / "logs"

        for dir_path in [self.tensorboard_dir, self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # Save configuration
        self._save_config()

    def _save_config(self):
        """Save all configurations to JSON"""
        config_dict = {
            "env_config": asdict(self.env_config),
            "ppo_config": asdict(self.ppo_config),
            "training_config": asdict(self.train_config),
            "timestamp": datetime.now().isoformat(),
        }

        config_path = self.results_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"✓ Configuration saved to: {config_path}")

    def create_model(self, env) -> PPO:
        """Create or load PPO model"""
        if self.resume_from:
            print(f"Loading model from: {self.resume_from}")
            model = PPO.load(self.resume_from, env=env)
            # Update learning rate if changed
            model.learning_rate = self.ppo_config.learning_rate
            return model

        # Create new model
        tensorboard_log = str(self.tensorboard_dir) if self.train_config.use_tensorboard else None

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.ppo_config.learning_rate,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=self.ppo_config.clip_range,
            ent_coef=self.ppo_config.ent_coef,
            vf_coef=self.ppo_config.vf_coef,
            clip_range_vf=self.ppo_config.clip_range_vf,
            max_grad_norm=self.ppo_config.max_grad_norm,
            seed=self.ppo_config.seed,
            device=self.ppo_config.device,
            verbose=self.train_config.verbose,
            tensorboard_log=tensorboard_log,
        )

        return model

    def create_callbacks(self, eval_env):
        """Create training callbacks"""
        callbacks = []

        # Episode logging callback
        self.episode_callback = CallbackPerEpisode(
            log_dir=str(self.logs_dir),
            max_stored_episodes=100,
            num_envs_to_track=self.train_config.num_envs_to_track,
        )
        callbacks.append(self.episode_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.train_config.save_freq,
            save_path=str(self.checkpoints_dir),
            name_prefix="ppo_guest",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.results_dir / "best_model"),
                log_path=str(self.logs_dir / "eval"),
                eval_freq=self.train_config.eval_freq,
                n_eval_episodes=self.train_config.eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        return CallbackList(callbacks)

    def train(self):
        """Execute training"""
        print("\n" + "=" * 70)
        print("STARTING PROFESSIONAL RL TRAINING")
        print("=" * 70)

        # Calculate total timesteps
        total_timesteps = self.env_config.max_steps * self.train_config.total_episodes

        print(f"\nConfiguration:")
        print(f"  Episodes: {self.train_config.total_episodes}")
        print(f"  Steps per episode: {self.env_config.max_steps}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Parallel envs: {self.train_config.num_parallel_envs}")
        print(f"  Learning rate: {self.ppo_config.learning_rate}")
        print(f"  Batch size: {self.ppo_config.batch_size}")
        print(f"  Results dir: {self.results_dir}")

        # Create environments
        train_env = create_vectorized_env(
            self.env_config,
            num_envs=self.train_config.num_parallel_envs,
            use_subprocess=self.train_config.num_parallel_envs > 1,
        )

        # Create evaluation environment (single env)
        eval_env = create_vectorized_env(self.env_config, num_envs=1, use_subprocess=False)

        # Create model
        model = self.create_model(train_env)

        # Create callbacks
        callbacks = self.create_callbacks(eval_env)

        print("\n" + "-" * 70)
        print("Training started...")
        if self.train_config.use_tensorboard:
            print(f"TensorBoard: tensorboard --logdir {self.tensorboard_dir}")
        print("-" * 70 + "\n")

        # Train!
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name="PPO_GuestEnv",
                progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user!")

        # Save final model
        final_model_path = self.results_dir / f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")

        # Save callback data to pickle file for graphs
        if self.train_config.save_pkl:
            pkl_filename = f"train_{datetime.now().strftime('%Y%m%d')}.pkl"
            pkl_path = self.results_dir / pkl_filename
            self.episode_callback.save_data(str(pkl_path))
            print(f"✓ Training data saved to: {pkl_path}")
            print(f"  ({self.train_config.num_envs_to_track} environment(s) tracked)")

        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Results directory: {self.results_dir}")
        print(f"Best model: {self.results_dir / 'best_model'}")
        print(f"Checkpoints: {self.checkpoints_dir}")
        print(f"Logs: {self.logs_dir}")

        # Clean up training environments
        train_env.close()
        eval_env.close()

        # Test the trained policy
        if self.train_config.test_after_training:
            test_results = self.test_policy(
                num_test_episodes=self.train_config.test_episodes,
                deterministic=True,
            )

        return model

    def test_policy(
        self,
        model_path: Optional[Path] = None,
        num_test_episodes: int = 50,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Test the trained policy comprehensively

        Args:
            model_path: Path to model to test (default: best_model)
            num_test_episodes: Number of test episodes
            deterministic: Use deterministic policy

        Returns:
            Dictionary with test results
        """
        print("\n" + "=" * 70)
        print("TESTING TRAINED POLICY")
        print("=" * 70)

        # Determine which model to test
        if model_path is None:
            model_path = self.results_dir / "best_model" / "best_model.zip"
            if not model_path.exists():
                # Fallback to final model
                final_models = list(self.results_dir.glob("final_model_*"))
                if final_models:
                    model_path = final_models[-1]
                else:
                    print("❌ No model found to test!")
                    return {}

        print(f"Testing model: {model_path}")
        print(f"Test episodes: {num_test_episodes}")
        print(f"Deterministic: {deterministic}")

        # Create test environment
        test_env = create_vectorized_env(self.env_config, num_envs=1, use_subprocess=False)

        # Load model
        model = PPO.load(model_path)

        # Run test episodes
        episode_rewards = []
        episode_ginis = []
        episode_lengths = []
        episode_phonemes = []
        action_counts = np.zeros(7)  # 7 actions

        print("\nRunning test episodes...")
        for episode in range(num_test_episodes):
            obs = test_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_gini_sum = 0
            gini_count = 0

            while not done:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = test_env.step(action)

                episode_reward += reward[0]
                episode_length += 1
                action_counts[action[0]] += 1

                # Track gini if available
                if 'current_gini' in info[0]:
                    episode_gini_sum += info[0]['current_gini']
                    gini_count += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if gini_count > 0:
                episode_ginis.append(episode_gini_sum / gini_count)

            if 'phoneme' in info[0]:
                episode_phonemes.append(info[0]['phoneme'].copy())

            # Progress indicator
            if (episode + 1) % 10 == 0:
                print(f"  Completed {episode + 1}/{num_test_episodes} episodes...")

        # Calculate statistics
        results = {
            'num_episodes': num_test_episodes,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'action_distribution': (action_counts / action_counts.sum()).tolist(),
            'action_counts': action_counts.tolist(),
        }

        if episode_ginis:
            results['mean_gini'] = float(np.mean(episode_ginis))
            results['std_gini'] = float(np.std(episode_ginis))
            results['min_gini'] = float(np.min(episode_ginis))
            results['max_gini'] = float(np.max(episode_ginis))

        if episode_phonemes:
            # Calculate phoneme balance statistics
            phoneme_array = np.array(episode_phonemes)
            mean_phonemes = phoneme_array.mean(axis=0)
            results['mean_phonemes'] = mean_phonemes.tolist()
            results['phoneme_std'] = phoneme_array.std(axis=0).tolist()

            # Calculate balance metric (coefficient of variation)
            phoneme_totals = phoneme_array.sum(axis=1)
            phoneme_stds = phoneme_array.std(axis=1)
            balance_metric = phoneme_stds / (phoneme_totals / 3)
            results['balance_metric_mean'] = float(np.mean(balance_metric))
            results['balance_metric_std'] = float(np.std(balance_metric))

        # Save results
        results_path = self.results_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print results
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"\n📊 Performance Metrics:")
        print(f"  Mean reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Reward range:    [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"  Mean ep length:  {results['mean_length']:.1f} steps")

        if 'mean_gini' in results:
            print(f"\n⚖️  Balance Metrics:")
            print(f"  Mean Gini:       {results['mean_gini']:.4f} ± {results['std_gini']:.4f}")
            print(f"  Gini range:      [{results['min_gini']:.4f}, {results['max_gini']:.4f}]")
            print(f"  (Lower is better - 0.0 = perfect balance)")

        if 'mean_phonemes' in results:
            print(f"\n🗣️  Phoneme Distribution:")
            for i, count in enumerate(results['mean_phonemes']):
                print(f"  Agent {i}:        {count:.1f} phonemes/episode")
            print(f"\n  Balance metric:  {results['balance_metric_mean']:.4f} ± {results['balance_metric_std']:.4f}")
            print(f"  (Lower is better - 0.0 = perfect balance)")

        print(f"\n🎯 Action Distribution:")
        actions = ['wait', 'stare_0', 'stare_1', 'stare_2', 'encourage_0', 'encourage_1', 'encourage_2']
        for i, (action, pct) in enumerate(zip(actions, results['action_distribution'])):
            bar = '█' * int(pct * 50)
            print(f"  {action:12s}: {bar:50s} {pct*100:5.1f}%")

        print(f"\n✅ Test results saved to: {results_path}")
        print("=" * 70)

        test_env.close()
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Professional RL Training for GuestEnv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        "--test-only",
        type=str,
        metavar="MODEL_PATH",
        help="Test a trained model without training (provide path to .zip model file)"
    )

    # Preset or custom config
    parser.add_argument(
        "--preset",
        type=str,
        choices=["dev", "fast", "production", "default"],
        default="default",
        help="Use a configuration preset"
    )

    # Training arguments
    parser.add_argument("--episodes", type=int, help="Total episodes to train")
    parser.add_argument("--parallel-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--envs-to-track", type=int, help="Number of envs to save to pkl (default: same as parallel-envs)")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Paths
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Results directory (default: auto-generated)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to model to resume training from"
    )

    # Flags
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--no-pkl", action="store_true", help="Disable saving pkl file for graphs")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable reward shaping")

    # Testing arguments
    parser.add_argument("--no-test", action="store_true", help="Skip testing after training")
    parser.add_argument("--test-episodes", type=int, help="Number of test episodes (default: 50)")

    args = parser.parse_args()

    # Handle test-only mode
    if args.test_only:
        test_episodes = args.test_episodes if args.test_episodes else 50
        test_only(
            model_path=args.test_only,
            test_episodes=test_episodes,
            preset=args.preset,
        )
        return

    # Load preset configuration
    env_config, ppo_config, train_config = get_config_preset(args.preset)

    # Override with command line arguments
    if args.episodes:
        train_config.total_episodes = args.episodes
    if args.parallel_envs:
        train_config.num_parallel_envs = args.parallel_envs
        # By default, track same number of envs as parallel envs (unless overridden)
        if not args.envs_to_track:
            train_config.num_envs_to_track = args.parallel_envs
    if args.envs_to_track:
        train_config.num_envs_to_track = args.envs_to_track
    if args.learning_rate:
        ppo_config.learning_rate = args.learning_rate
    if args.batch_size:
        ppo_config.batch_size = args.batch_size
    if args.seed:
        env_config.seed = args.seed
        ppo_config.seed = args.seed
    if args.no_tensorboard:
        train_config.use_tensorboard = False
    if args.no_pkl:
        train_config.save_pkl = False
    if args.reward_shaping:
        env_config.reward_shaping = True
    if args.no_test:
        train_config.test_after_training = False
    if args.test_episodes:
        train_config.test_episodes = args.test_episodes

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("src/test_result/result") / f"train_{args.preset}_{timestamp}"

    # Create trainer and run
    trainer = Trainer(
        env_config=env_config,
        ppo_config=ppo_config,
        train_config=train_config,
        results_dir=results_dir,
        resume_from=args.resume_from,
    )

    trainer.train()


def test_only(model_path: str, test_episodes: int = 50, preset: str = "default"):
    """
    Test a trained model without training

    Args:
        model_path: Path to saved model (.zip file)
        test_episodes: Number of test episodes
        preset: Environment preset to use
    """
    print("\n" + "=" * 70)
    print("STANDALONE POLICY TESTING")
    print("=" * 70)

    # Load configuration
    env_config, ppo_config, train_config = get_config_preset(preset)

    # Create results directory for test
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ Model not found: {model_path}")
        return

    results_dir = model_path_obj.parent.parent / "test_only_results"
    results_dir.mkdir(exist_ok=True)

    # Create trainer just for testing
    trainer = Trainer(
        env_config=env_config,
        ppo_config=ppo_config,
        train_config=train_config,
        results_dir=results_dir,
    )

    # Run test
    trainer.test_policy(
        model_path=model_path_obj,
        num_test_episodes=test_episodes,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
