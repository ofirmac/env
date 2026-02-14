import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.env_gym import GuestEnv
from src.callback.guest_callback_per_episode_multi_env import CallbackPerEpisode

MAX_STEPS = 500
TOTAL_TIMESTEPS = MAX_STEPS*100

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessing policy.
    """
    def _init():
        env = GuestEnv(
            max_steps=MAX_STEPS, 
            reward_shaping=False,
            logfile=False,  # Disable logging for performance
            seed=seed + rank
        )
        
        # Configure agent parameters
        env.agent_params[0].update({  # quiet analyst
            "min_energy_to_speak": 0.20,
            "energy_gain": 0.002,
            "energy_decay": 0.08,
            "max_speaking_time": 6,
            "phonemes_per_step": 4,
        })

        env.agent_params[1].update({  # balanced mediator
            "min_energy_to_speak": 0.55,
            "energy_gain": 0.010,
            "energy_decay": 0.05,
            "max_speaking_time": 8,
            "phonemes_per_step": 5,
        })

        env.agent_params[2].update({  # energetic storyteller
            "min_energy_to_speak": 0.85,
            "energy_gain": 0.028,
            "energy_decay": 0.03,
            "max_speaking_time": 12,
            "phonemes_per_step": 6,
        })
        
        return env
    
    set_random_seed(seed)
    return _init

def train_optimized(total_timesteps=TOTAL_TIMESTEPS, results_dir: str = None, use_multiprocessing: bool = True, num_envs: int = 4) -> None:
    """
    Optimized training function with performance improvements.
    
    Args:
        total_timesteps: Total training timesteps
        results_dir: Directory to save results
        use_multiprocessing: Whether to use multiple parallel environments
        num_envs: Number of parallel environments (if multiprocessing enabled)
    """
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "ppo_results_optimized")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create vectorized environment for better performance
    if use_multiprocessing and num_envs > 1:
        print(f"Using {num_envs} parallel environments for faster training...")
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    else:
        print("Using single environment...")
        # Single environment setup
        single_env = GuestEnv(max_steps=MAX_STEPS, reward_shaping=False, logfile=False)
        
        single_env.agent_params[0].update({
            "min_energy_to_speak": 0.20,
            "energy_gain": 0.002,
            "energy_decay": 0.08,
            "max_speaking_time": 6,
            "phonemes_per_step": 4,
        })

        single_env.agent_params[1].update({
            "min_energy_to_speak": 0.55,
            "energy_gain": 0.010,
            "energy_decay": 0.05,
            "max_speaking_time": 8,
            "phonemes_per_step": 5,
        })

        single_env.agent_params[2].update({
            "min_energy_to_speak": 0.85,
            "energy_gain": 0.028,
            "energy_decay": 0.03,
            "max_speaking_time": 12,
            "phonemes_per_step": 6,
        })
        
        env = DummyVecEnv([lambda: single_env])
    
    # Optimized PPO configuration for better performance
    tensorboard_log = os.path.join(results_dir, "tensorboard")
    
    # PERFORMANCE OPTIMIZATIONS:
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,           # Standard learning rate
        n_steps=2048,                 # REDUCED from MAX_STEPS (500) - much more efficient
        batch_size=256,               # REDUCED from MAX_STEPS - faster gradient computation
        n_epochs=4,                   # REDUCED from 5 - less overfitting, faster training
        gamma=0.99,                   # Standard discount factor
        gae_lambda=0.95,
        clip_range=0.2,               # Standard clipping
        ent_coef=0.01,                # Standard entropy coefficient
        vf_coef=0.5,                  # REDUCED - less critic dominance, faster training
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        seed=42,
        verbose=1,
        device='auto',                # Let PyTorch choose best device
    )

    # Lightweight callback for better performance
    callback = CallbackPerEpisode(
        log_dir=os.path.join(results_dir, "detailed_logs"),
        max_stored_episodes=50        # REDUCED from 100 - less memory usage
    )
    
    print(f"Starting OPTIMIZED training for {total_timesteps} timesteps...")
    print(f"Configuration:")
    print(f"  - Parallel environments: {num_envs if use_multiprocessing else 1}")
    print(f"  - n_steps: 2048 (reduced from {MAX_STEPS})")
    print(f"  - batch_size: 256 (reduced from {MAX_STEPS})")
    print(f"  - n_epochs: 4 (reduced from 5)")
    print(f"  - Max stored episodes: 50 (reduced from 100)")
    print(f"TensorBoard logs: {tensorboard_log}")
    print(f"Detailed logs: {os.path.join(results_dir, 'detailed_logs')}")
    
    # Train the model
    total_timesteps = int(total_timesteps)
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback,
        tb_log_name="PPO_GuestEnv_Optimized"
    )
    
    # Save the model
    date_str = datetime.now().strftime("%Y%m%d")    
    model.save(os.path.join(results_dir, f"ppo_guest_optimized_{date_str}"))

    # Save callback data
    callback.save_data(f"{results_dir}/train_optimized_{date_str}.pkl")

    print(f"Training completed! Results saved to: {results_dir}")
    print(f"Total episodes: {callback.episode_count}")
    if callback.episode_rewards:
        print(f"Average episode reward: {np.mean(callback.episode_rewards):.3f}")
        print(f"Final episode reward: {(callback.episode_rewards[-1]/MAX_STEPS):.3f}")

def train_ultra_fast(total_timesteps=TOTAL_TIMESTEPS, results_dir: str = None) -> None:
    """
    Ultra-fast training with minimal logging for maximum performance.
    """
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "ppo_results_ultrafast")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Single environment with minimal overhead
    env = GuestEnv(max_steps=MAX_STEPS, reward_shaping=False, logfile=False)
    
    # Configure agent parameters
    env.agent_params[0].update({
        "min_energy_to_speak": 0.20,
        "energy_gain": 0.002,
        "energy_decay": 0.08,
        "max_speaking_time": 6,
        "phonemes_per_step": 4,
    })

    env.agent_params[1].update({
        "min_energy_to_speak": 0.55,
        "energy_gain": 0.010,
        "energy_decay": 0.05,
        "max_speaking_time": 8,
        "phonemes_per_step": 5,
    })

    env.agent_params[2].update({
        "min_energy_to_speak": 0.85,
        "energy_gain": 0.028,
        "energy_decay": 0.03,
        "max_speaking_time": 12,
        "phonemes_per_step": 6,
    })
    
    env = DummyVecEnv([lambda: env])
    
    # Ultra-fast PPO configuration
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,                 # Even smaller for speed
        batch_size=128,               # Smaller batch for speed
        n_epochs=3,                   # Fewer epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=None,         # NO TENSORBOARD for max speed
        seed=42,
        verbose=1,
        device='auto',
    )

    print(f"Starting ULTRA-FAST training for {total_timesteps} timesteps...")
    print("Configuration: Minimal logging, no TensorBoard, optimized batch sizes")
    
    # Train without callback for maximum speed
    model.learn(total_timesteps=total_timesteps)
    
    # Save only the model
    date_str = datetime.now().strftime("%Y%m%d")    
    model.save(os.path.join(results_dir, f"ppo_guest_ultrafast_{date_str}"))
    print(f"Ultra-fast training completed! Model saved to: {results_dir}")

if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y_%m_%d")
    print(f"{date_str}")
    
    # Choose your training mode:
    
    # Option 1: Optimized training with parallel environments (RECOMMENDED)
    results_dir = os.path.join(os.getcwd(), "src", "test_result", f"train_optimized_{date_str}")
    train_optimized(results_dir=results_dir, use_multiprocessing=True, num_envs=4)
    
    # Option 2: Ultra-fast training with minimal logging (uncomment to use)
    # results_dir = os.path.join(os.getcwd(), "src", "test_result", f"train_ultrafast_{date_str}")
    # train_ultra_fast(results_dir=results_dir)