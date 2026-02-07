import os
import sys
import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.env_gym import GuestEnv
from src.callback.guest_callback import TensorBoardMetricsCallback
from src.callback.guest_callback_per_episode import CallbackPerEpisode

MAX_STEPS = 500
MAX_EPISODE = 5000
# MAX_EPISODE = 2
TOTAL_TIMESTEPS = MAX_STEPS*MAX_EPISODE

#-------    Main training function -------
def train(total_timesteps=TOTAL_TIMESTEPS, results_dir: str = os.path.join(os.getcwd(), "ppo_results_per_episode.pkl")) -> None:
    """Enhanced training function with comprehensive logging."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Create environment
    env = GuestEnv(max_steps=MAX_STEPS, reward_shaping=False, env_effect=False, efficiency=False, randomize_agent=False ,seed=42)


    env.agent_params[0].update({  # quiet analyst
    "min_energy_to_speak": 0.20,
    "energy_gain": 0.0018,
    "energy_decay": 0.09,
    "max_speaking_time": 500,
    "phonemes_per_step": 4,
    })

    env.agent_params[1].update({  # balanced mediator
        "min_energy_to_speak": 0.55,
        "energy_gain": 0.010,
        "energy_decay": 0.05,
        "max_speaking_time": 500,
        "phonemes_per_step": 5,
    })

    env.agent_params[2].update({  # energetic storyteller
        "min_energy_to_speak": 0.85,
        "energy_gain": 0.030,
        "energy_decay": 0.025,
        "max_speaking_time": 500,
        "phonemes_per_step": 6,
    })

    # env.agent_params[0].update({  # quiet analyst
    # "min_energy_to_speak": 0.20,
    # "energy_gain": 0.002,
    # "energy_decay": 0.08,
    # "max_speaking_time": 6,
    # "phonemes_per_step": 4,
    # })

    # env.agent_params[1].update({  # balanced mediator
    #     "min_energy_to_speak": 0.55,
    #     "energy_gain": 0.010,
    #     "energy_decay": 0.05,
    #     "max_speaking_time": 8,
    #     "phonemes_per_step": 5,
    # })

    # env.agent_params[2].update({  # energetic storyteller
    #     "min_energy_to_speak": 0.85,
    #     "energy_gain": 0.028,
    #     "energy_decay": 0.03,
    #     "max_speaking_time": 12,
    #     "phonemes_per_step": 6,
    # })
    env_config = {
        "reward_shaping": False,
        "env_effect": False,
        "efficiency": False,
        "randomize_agent": False,
        "agent_params": env.agent_params,  # list of dicts
    }
    print(env.agent_params[2])
    env = DummyVecEnv([lambda: env])
    
    # Create model with TensorBoard logging
    tensorboard_log = os.path.join(results_dir, "tensorboard")
    # model = PPO(
    #     "MlpPolicy", 
    #     env, 
    #     verbose=1,
    #     tensorboard_log=tensorboard_log,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01
    # )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,         # stronger updates
        n_steps=MAX_STEPS,
        batch_size=MAX_STEPS,         # full-episode batch
        n_epochs=5,                 # fewer epochs (avoid overfitting critic)
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.25,             # looser clipping
        ent_coef=0.01,             # weaker entropy
        vf_coef=0.5,                # reduce critic dominance
        clip_range_vf=0.2,          # clip critic updates
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        seed=42,
        verbose=1,
        device='cpu'
    )

    ppo_config = {
        "policy": "MlpPolicy",
        "learning_rate": 5e-4,
        "n_steps": MAX_STEPS,
        "batch_size": MAX_STEPS,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.25,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "clip_range_vf": 0.2,
        "max_grad_norm": 0.5,
        "tensorboard_log": tensorboard_log,
        "seed": 42,
        "verbose": 1,
    }

    metadata = {
        "MAX_STEPS": MAX_STEPS,
        "MAX_EPISODE": MAX_EPISODE,
        "TOTAL_TIMESTEPS": total_timesteps,
        "env_config": env_config,
        "ppo_config": ppo_config,
        "timestamp": datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(results_dir, "run_metadata.json")
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Saved run metadata to: {metadata_path}")
    except Exception as e:
        print(f"[WARN] Failed to save metadata JSON: {e}")

    callback = CallbackPerEpisode(
        log_dir=os.path.join(results_dir, "detailed_logs")
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"TensorBoard logs will be saved to: {tensorboard_log}")
    print(f"Detailed logs will be saved to: {os.path.join(results_dir, 'detailed_logs')}")
    print(f"Run 'tensorboard --logdir {results_dir}' to view logs")
    
    # Train the model
    total_timesteps = int(total_timesteps)
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback,
        tb_log_name="PPO_GuestEnv"
    )

    date_str = datetime.now().strftime("%Y%m%d") 
    try:
        model.save(os.path.join(results_dir, f"ppo_guest_train_{date_str}"))
    except Exception as e:
        print(f"[WARN] Model save failed: {e}")
        # Still persist per-episode data so you can plot later
    finally:
        callback.save_data(f"{results_dir}/train_{date_str}.pkl")

    # # Save the model
    # date_str = datetime.now().strftime("%Y%m%d")    
    # model.save(os.path.join(results_dir, f"ppo_guest_train_{date_str}"))

    # # Plot all episodes (if you have few episodes)
    # callback.save_data(f"{results_dir}/train_{date_str}.pkl")

    print(f"Training completed! Results saved to: {results_dir}")
    print(f"Total episodes: {callback.episode_count}") 
    if callback.episode_rewards:
        avg = [i/MAX_STEPS for i in callback.episode_rewards]
        print(f"Average episode reward: {np.mean(avg):.3f}")
        print(f"Final episode reward: {(callback.episode_rewards[-1]/MAX_STEPS):.3f}")

if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y_%m_%d")
    print(f"{date_str}")
    # Use current working directory for results
    results_dir_name = f"train_result_{date_str}_{MAX_STEPS=}_{MAX_EPISODE=}_500_per_step_moshe"
    os.environ['RESULTS_DIR'] = results_dir_name
    results_dir = os.path.join(os.getcwd(), "src", "test_result", results_dir_name)
    train(results_dir=results_dir)