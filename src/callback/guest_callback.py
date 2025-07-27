import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class TensorBoardMetricsCallback(BaseCallback):
    """Enhanced callback for recording detailed metrics with TensorBoard."""
    
    def __init__(self, log_dir: str = "./tensorboard_logs/"):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        
        # Episode-level tracking
        self.episode_rewards = []
        self.episode_phonemes = []
        self.episode_gini = []
        self.episode_actions = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.episode_count = 0
        
        # Step-level tracking for detailed analysis
        self.step_rewards = []
        self.step_phonemes = []
        self.step_gini = []
        self.step_actions = []
        
    def _on_training_start(self) -> None:
        """Initialize TensorBoard writer."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Get info from the environment
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0])
        
        if len(infos) > 0 and len(rewards) > 0:
            info = infos[0]
            reward = rewards[0]
            
            # Track current episode
            self.current_episode_reward += reward
            self.current_episode_steps += 1
            
            # Get metrics from info
            phonemes = info.get("phoneme", [0, 0, 0])
            gini_history = info.get("gini_history", [])
            current_gini = gini_history[-1] if gini_history else 0
            action_number = info.get("action_number", -1)
            env_reward = info.get("env_reward", 0)
            
            # Log step-level metrics
            global_step = self.num_timesteps
            
            # Rewards
            self.writer.add_scalar('Step/Reward', reward, global_step)
            self.writer.add_scalar('Step/EnvReward', env_reward, global_step)
            
            # Phonemes per agent
            for i, phoneme_count in enumerate(phonemes):
                self.writer.add_scalar(f'Step/Agent_{i}_Phonemes', phoneme_count, global_step)
            
            # Gini coefficient
            self.writer.add_scalar('Step/Gini_Coefficient', current_gini, global_step)
            
            # Actions
            if action_number >= 0:
                self.writer.add_scalar('Step/Action', action_number, global_step)
            
            # Phoneme distribution
            total_phonemes = sum(phonemes)
            if total_phonemes > 0:
                for i, phoneme_count in enumerate(phonemes):
                    percentage = (phoneme_count / total_phonemes) * 100
                    self.writer.add_scalar(f'Step/Agent_{i}_Percentage', percentage, global_step)
            
            # Check if episode ended
            dones = self.locals.get("dones", [False])
            if dones[0]:  # Episode ended
                self._log_episode_metrics(info)
                
        return True
    
    def _log_episode_metrics(self, final_info):
        """Log episode-level metrics."""
        self.episode_count += 1
        
        # Episode reward
        self.episode_rewards.append(self.current_episode_reward)
        self.writer.add_scalar('Episode/Total_Reward', self.current_episode_reward, self.episode_count)
        self.writer.add_scalar('Episode/Average_Reward', self.current_episode_reward / self.current_episode_steps, self.episode_count)
        
        # Final phoneme counts
        final_phonemes = final_info.get("phoneme", [0, 0, 0])
        self.episode_phonemes.append(final_phonemes.copy())
        
        for i, phoneme_count in enumerate(final_phonemes):
            self.writer.add_scalar(f'Episode/Agent_{i}_Final_Phonemes', phoneme_count, self.episode_count)
        
        # Episode balance metrics
        total_phonemes = sum(final_phonemes)
        if total_phonemes > 0:
            # Phoneme percentages
            percentages = [(count / total_phonemes) * 100 for count in final_phonemes]
            for i, percentage in enumerate(percentages):
                self.writer.add_scalar(f'Episode/Agent_{i}_Final_Percentage', percentage, self.episode_count)
            
            # Balance metrics
            std_dev = np.std(final_phonemes)
            self.writer.add_scalar('Episode/Phoneme_StdDev', std_dev, self.episode_count)
            
            # Gini coefficient
            gini_history = final_info.get("gini_history", [])
            if gini_history:
                final_gini = gini_history[-1]
                self.episode_gini.append(final_gini)
                self.writer.add_scalar('Episode/Final_Gini', final_gini, self.episode_count)
        
        # Action statistics
        action_stats = final_info.get("actions_stats", [])
        if len(action_stats) > 0:
            for i, action_count in enumerate(action_stats):
                self.writer.add_scalar(f'Episode/Action_{i}_Count', action_count, self.episode_count)
        
        # Episode length
        self.writer.add_scalar('Episode/Length', self.current_episode_steps, self.episode_count)
        
        # Reset for next episode
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
        # Log episode summary every 10 episodes
        if self.episode_count % 10 == 0:
            self._log_summary_statistics()
    
    def _log_summary_statistics(self):
        """Log summary statistics over recent episodes."""
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            recent_phonemes = self.episode_phonemes[-10:]
            
            # Average metrics over last 10 episodes
            avg_reward = np.mean(recent_rewards)
            self.writer.add_scalar('Summary/Avg_Episode_Reward_10', avg_reward, self.episode_count)
            
            # Average phoneme distribution
            avg_phonemes = np.mean(recent_phonemes, axis=0)
            for i, avg_count in enumerate(avg_phonemes):
                self.writer.add_scalar(f'Summary/Avg_Agent_{i}_Phonemes_10', avg_count, self.episode_count)
            
            # Balance over last 10 episodes
            phoneme_std = np.std(avg_phonemes)
            self.writer.add_scalar('Summary/Avg_Balance_StdDev_10', phoneme_std, self.episode_count)
    
    def _on_training_end(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def create_final_plots(self, results_dir: str):
        """Create final matplotlib plots for the paper."""
        os.makedirs(results_dir, exist_ok=True)
        
        # Episode rewards plot
        if self.episode_rewards:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Episode Rewards During Training')
            plt.grid(True, alpha=0.3)
            
            # Moving average
            if len(self.episode_rewards) > 10:
                window = min(50, len(self.episode_rewards) // 10)
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', alpha=0.7, label=f'Moving Avg ({window})')
                plt.legend()
            
            plt.subplot(1, 2, 2)
            if self.episode_phonemes:
                phonemes_array = np.array(self.episode_phonemes)
                for i in range(phonemes_array.shape[1]):
                    plt.plot(phonemes_array[:, i], label=f'Agent {i}', alpha=0.7)
                plt.xlabel('Episode')
                plt.ylabel('Final Phoneme Count')
                plt.title('Agent Phonemes per Episode')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'episode_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Balance analysis
        if self.episode_phonemes:
            plt.figure(figsize=(10, 6))
            
            phonemes_array = np.array(self.episode_phonemes)
            
            # Calculate balance metrics per episode
            gini_coeffs = []
            std_devs = []
            
            for episode_phonemes in phonemes_array:
                total = np.sum(episode_phonemes)
                if total > 0:
                    # Gini coefficient
                    x = episode_phonemes.astype(float)
                    diffs = np.abs(x[:, None] - x[None, :]).sum()
                    n = len(x)
                    gini = float(diffs / (2 * n * total))
                    gini_coeffs.append(gini)
                    
                    # Standard deviation
                    std_devs.append(np.std(episode_phonemes))
                else:
                    gini_coeffs.append(0)
                    std_devs.append(0)
            
            plt.subplot(2, 1, 1)
            plt.plot(gini_coeffs)
            plt.xlabel('Episode')
            plt.ylabel('Gini Coefficient')
            plt.title('Conversation Balance (Lower = More Equal)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(std_devs)
            plt.xlabel('Episode')
            plt.ylabel('Standard Deviation')
            plt.title('Phoneme Distribution Variability')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'balance_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()