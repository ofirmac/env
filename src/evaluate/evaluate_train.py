import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.env_gym import GuestEnv

#----- End of the training script -----
def evaluate_and_plot(model_path: str, results_dir: str, n_episodes: int = 100):
    """Evaluate trained model and create publication-ready plots."""
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = GuestEnv(max_steps=MAX_STEPS, reward_shaping=True)
    
    # Run evaluation episodes
    episode_rewards = []
    episode_phonemes = []
    episode_gini = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_phonemes.append(info['phoneme'].copy())
        if info['gini_history']:
            episode_gini.append(info['gini_history'][-1])
    
    # Create publication plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Evaluation Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Phoneme distribution
    phonemes_array = np.array(episode_phonemes)
    for i in range(3):
        axes[0, 1].plot(phonemes_array[:, i], label=f'Agent {i}', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Final Phoneme Count')
    axes[0, 1].set_title('Agent Speaking Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gini coefficient
    axes[1, 0].plot(episode_gini)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Gini Coefficient')
    axes[1, 0].set_title('Conversation Inequality (Lower = Better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final distribution histogram
    final_percentages = []
    for phonemes in episode_phonemes:
        total = sum(phonemes)
        if total > 0:
            percentages = [(count/total)*100 for count in phonemes]
            final_percentages.append(percentages)
    
    if final_percentages:
        final_percentages = np.array(final_percentages)
        x = np.arange(3)
        width = 0.25
        
        means = np.mean(final_percentages, axis=0)
        stds = np.std(final_percentages, axis=0)
        
        bars = axes[1, 1].bar(x, means, width, yerr=stds, capsize=5)
        axes[1, 1].set_xlabel('Agent')
        axes[1, 1].set_ylabel('Speaking Percentage (%)')
        axes[1, 1].set_title('Average Speaking Distribution')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'Agent {i}' for i in range(3)])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                           f'{mean:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average Gini coefficient: {np.mean(episode_gini):.3f} ± {np.std(episode_gini):.3f}")
    
    if final_percentages is not None:
        print("\nSpeaking distribution:")
        for i in range(3):
            mean_pct = np.mean(final_percentages[:, i])
            std_pct = np.std(final_percentages[:, i])
            print(f"Agent {i}: {mean_pct:.1f}% ± {std_pct:.1f}%")