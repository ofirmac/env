import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.env_new_speaker.test_new_speaker_env import GuestEnv
from callback.guest_callback import TensorBoardMetricsCallback
from src.callback.guest_callback_per_episode_multi_env import CallbackPerEpisode


MAX_STEPS = 100
TOTAL_TIMESTEPS = MAX_STEPS*700

#-------    Main training function -------
def train(total_timesteps=TOTAL_TIMESTEPS, results_dir: str = "ppo_results_per_episode_pkl") -> None:
    """Enhanced training function with comprehensive logging."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Create environment
    env = GuestEnv(max_steps=MAX_STEPS, reward_shaping=False)  # Shorter episodes for more frequent logging
    
    # Update agent parameters
    env.agent_params[0].update({
        'min_energy_to_speak': 0.6,
        'energy_gain': 0.01,
        'energy_decay': 0.15,
        'max_speaking_time': 2,
        'phonemes_per_step': 1
    })
    env.agent_params[1].update({
        'min_energy_to_speak': 0.3,
        'energy_gain': 0.05,
        'energy_decay': 0.10,
        'max_speaking_time': 5,
        'phonemes_per_step': 2
    })
    env.agent_params[2].update({
        'min_energy_to_speak': 0.1,
        'energy_gain': 0.10,
        'energy_decay': 0.05,
        'max_speaking_time': 8,
        'phonemes_per_step': 4
    })
    
    env = DummyVecEnv([lambda: env])
    
    # Create model with TensorBoard logging
    tensorboard_log = os.path.join(results_dir, "tensorboard")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Create enhanced callback
    # callback = TensorBoardMetricsCallback(
    #     log_dir=os.path.join(results_dir, "detailed_logs")
    # )
    callback = CallbackPerEpisode(
        log_dir=os.path.join(results_dir, "detailed_logs")
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"TensorBoard logs will be saved to: {tensorboard_log}")
    print(f"Detailed logs will be saved to: {os.path.join(results_dir, 'detailed_logs')}")
    print(f"Run 'tensorboard --logdir {results_dir}' to view logs")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback,
        tb_log_name="PPO_GuestEnv"
    )
    
    # Save the model
    model.save(os.path.join(results_dir, "ppo_guest_softmax"))
    
    # Create final plots
    # callback.create_final_plots(results_dir)

    # # Plot first 5 episodes
    # callback.create_final_plots("./results", max_episodes=5)

    # # Plot specific episodes (e.g., episodes 1, 5, and 10)
    # callback.create_final_plots("./results", plot_episodes=[0, 4, 9])

    # Plot all episodes (if you have few episodes)
    callback.save_data("ppo_results_per_episode_pkl/old_obs_softmax.pkl")
    # callback.create_final_plots(results_dir, plot_episodes=list(range(len(callback.episodes_step_data))))

    print(f"Training completed! Results saved to: {results_dir}")
    print(f"Total episodes: {callback.episode_count}")
    if callback.episode_rewards:
        print(f"Average episode reward: {np.mean(callback.episode_rewards):.3f}")
        print(f"Final episode reward: {callback.episode_rewards[-1]:.3f}")

if __name__ == "__main__":
    train()