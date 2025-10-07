import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import pickle
from typing import Dict, List, Optional, Union, Any
import numpy.typing as npt
from loguru import logger


class CallbackPerEpisodeLightweight(BaseCallback):
    """
    Lightweight version of CallbackPerEpisode optimized for performance.
    
    Reduces logging frequency and memory usage for faster training.
    
    Args:
        log_dir: Directory for TensorBoard logs
        max_stored_episodes: Maximum episodes to store for plotting (memory limit)
        log_frequency: Log step-level metrics every N steps (default: 10 for performance)
        detailed_plotting: Whether to store detailed step data (disable for speed)
    """
    
    def __init__(self, 
                 log_dir: str = "./tensorboard_logs/", 
                 max_stored_episodes: int = 50,
                 log_frequency: int = 10,
                 detailed_plotting: bool = True) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        self.max_stored_episodes = max_stored_episodes
        self.log_frequency = log_frequency
        self.detailed_plotting = detailed_plotting
        
        # Episode-level tracking (essential)
        self.episode_rewards = []
        self.episode_phonemes = []
        self.episode_gini = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.episode_count = 0
        
        # Detailed step data (optional for performance)
        if self.detailed_plotting:
            self.episodes_step_data = []
            self.current_episode_data = {
                'rewards': [],
                'phonemes': [],
                'gini': [],
                'actions': [],
                'cumulative_reward': [],
                'energy': []
            }
        
    def _on_training_start(self) -> None:
        """Initialize TensorBoard writer."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
    def _on_step(self) -> bool:
        """Called after each environment step - OPTIMIZED VERSION."""
        # Get info from the environment
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0])
        
        if len(infos) > 0 and len(rewards) > 0:
            info = infos[0]
            reward = rewards[0]
            
            # Track current episode (always needed)
            self.current_episode_reward += reward
            self.current_episode_steps += 1
            
            # PERFORMANCE OPTIMIZATION: Only log every N steps
            should_log_step = (self.num_timesteps % self.log_frequency == 0)
            
            if should_log_step or self.detailed_plotting:
                # Get metrics from info with safe extraction
                phonemes = self._safe_get_metric(info, "phoneme", [0, 0, 0])
                gini_history = self._safe_get_metric(info, "gini_history", [])
                current_gini = gini_history[-1] if gini_history else 0
                action_number = self._safe_get_metric(info, "action_number", -1)
                env_reward = self._safe_get_metric(info, "env_reward", 0)
                energy = self._safe_get_metric(info, "energy", [0, 0, 0])
                
                # Store detailed step data only if enabled
                if self.detailed_plotting:
                    self.current_episode_data['rewards'].append(reward)
                    self.current_episode_data['phonemes'].append(phonemes.copy() if isinstance(phonemes, list) else list(phonemes))
                    self.current_episode_data['gini'].append(current_gini)
                    self.current_episode_data['actions'].append(action_number)
                    self.current_episode_data['cumulative_reward'].append(self.current_episode_reward)
                    self.current_episode_data['energy'].append(energy.copy() if isinstance(energy, list) else list(energy))
                
                # REDUCED TensorBoard logging for performance
                if should_log_step:
                    global_step = self.num_timesteps
                    
                    # Essential metrics only
                    self.writer.add_scalar('Step/Reward', reward, global_step)
                    self.writer.add_scalar('Step/Gini_Coefficient', current_gini, global_step)
                    
                    # Reduced phoneme logging (only totals, not per-agent)
                    total_phonemes = sum(phonemes) if isinstance(phonemes, (list, np.ndarray)) else 0
                    self.writer.add_scalar('Step/Total_Phonemes', total_phonemes, global_step)
            
            # Check if episode ended
            dones = self.locals.get("dones", [False])
            if dones[0]:  # Episode ended
                self._log_episode_metrics(info)
                
        return True
    
    def _safe_get_metric(self, info: Dict[str, Any], key: str, default_value: Any) -> Any:
        """Safely extract metrics from info dict with validation."""
        try:
            value = info.get(key, default_value)
            if key in ["phoneme", "energy"] and not isinstance(value, (list, np.ndarray)):
                return default_value
            return value
        except Exception:
            return default_value
    
    def _log_episode_metrics(self, final_info: Dict[str, Any]) -> None:
        """Log episode-level metrics - OPTIMIZED VERSION."""
        self.episode_count += 1
        
        # Memory management: limit stored episodes
        if self.detailed_plotting and len(self.episodes_step_data) >= self.max_stored_episodes:
            self.episodes_step_data.pop(0)  # Remove oldest
        
        # Store episode data for detailed plotting (if enabled)
        if self.detailed_plotting:
            self.episodes_step_data.append(self.current_episode_data.copy())
        
        # Essential episode metrics
        self.episode_rewards.append(self.current_episode_reward)
        self.writer.add_scalar('Episode/Total_Reward', self.current_episode_reward, self.episode_count)
        
        if self.current_episode_steps > 0:
            self.writer.add_scalar('Episode/Average_Reward', self.current_episode_reward / self.current_episode_steps, self.episode_count)
        
        # Final phoneme counts
        final_phonemes = self._safe_get_metric(final_info, "phoneme", [0, 0, 0])
        self.episode_phonemes.append(final_phonemes.copy() if isinstance(final_phonemes, list) else list(final_phonemes))
        
        # REDUCED logging: Only essential metrics
        total_phonemes = sum(final_phonemes) if isinstance(final_phonemes, (list, np.ndarray)) else 0
        self.writer.add_scalar('Episode/Total_Final_Phonemes', total_phonemes, self.episode_count)
        
        if total_phonemes > 0:
            # Gini coefficient
            gini_history = self._safe_get_metric(final_info, "gini_history", [])
            if gini_history:
                final_gini = gini_history[-1]
                self.episode_gini.append(final_gini)
                self.writer.add_scalar('Episode/Final_Gini', final_gini, self.episode_count)
        
        # Episode length
        self.writer.add_scalar('Episode/Length', self.current_episode_steps, self.episode_count)
        
        # Reset for next episode
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
        if self.detailed_plotting:
            self.current_episode_data = {
                'rewards': [],
                'phonemes': [],
                'gini': [],
                'actions': [],
                'cumulative_reward': [],
                'energy': []
            }
        
        # REDUCED summary logging frequency
        if self.episode_count % 20 == 0:  # Every 20 episodes instead of 10
            self._log_summary_statistics()
    
    def _log_summary_statistics(self) -> None:
        """Log summary statistics over recent episodes."""
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            
            # Only essential summary metrics
            avg_reward = np.mean(recent_rewards)
            self.writer.add_scalar('Summary/Avg_Episode_Reward_10', avg_reward, self.episode_count)
    
    def _on_training_end(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def create_final_plots(self, results_dir: str, plot_episodes: Optional[List[int]] = None, max_episodes: int = 5) -> None:
        """
        Create final matplotlib plots - LIGHTWEIGHT VERSION.
        
        Args:
            results_dir: Directory to save plots
            plot_episodes: List of specific episode numbers to plot
            max_episodes: Maximum number of episodes to plot (reduced default)
        """
        if not self.detailed_plotting:
            print("Detailed plotting was disabled for performance. Only summary plots available.")
            self._create_summary_plots(results_dir)
            return
            
        os.makedirs(results_dir, exist_ok=True)
        
        # Create episode plots directory
        episodes_dir = os.path.join(results_dir, 'individual_episodes')
        os.makedirs(episodes_dir, exist_ok=True)
        
        # Determine which episodes to plot (fewer for performance)
        if plot_episodes is None:
            plot_episodes = list(range(min(len(self.episodes_step_data), max_episodes)))
        else:
            plot_episodes = [ep for ep in plot_episodes if 0 <= ep < len(self.episodes_step_data)]
        
        # Create individual episode plots
        for episode_idx in plot_episodes:
            if episode_idx < len(self.episodes_step_data):
                self._create_episode_plot_lightweight(episode_idx, episodes_dir)
        
        # Create summary plots
        self._create_summary_plots(results_dir)
        
        print(f"Created lightweight plots for {len(plot_episodes)} episodes in {episodes_dir}")
        print(f"Created summary plots in {results_dir}")
    
    def _create_episode_plot_lightweight(self, episode_idx: int, save_dir: str) -> None:
        """Create a lightweight plot for a single episode."""
        if episode_idx >= len(self.episodes_step_data):
            return
            
        episode_data = self.episodes_step_data[episode_idx]
        
        if not episode_data or not episode_data.get('rewards'):
            return
            
        steps = range(len(episode_data['rewards']))
        
        # SIMPLIFIED plotting - only essential plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Episode {episode_idx + 1} - Key Metrics', fontsize=14)
        
        # Plot 1: Rewards
        axes[0, 0].plot(steps, episode_data['rewards'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Rewards per Step')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Phonemes
        phonemes_array = np.array(episode_data['phonemes'])
        if len(phonemes_array) > 0 and phonemes_array.ndim > 1:
            for agent_idx in range(min(phonemes_array.shape[1], 3)):
                axes[0, 1].plot(steps, phonemes_array[:, agent_idx], label=f'Agent {agent_idx}')
        axes[0, 1].set_title('Phonemes per Agent')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Phoneme Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gini coefficient
        axes[1, 0].plot(steps, episode_data['gini'], 'g-', alpha=0.7)
        axes[1, 0].set_title('Balance (Gini Coefficient)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gini Coefficient')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative reward
        axes[1, 1].plot(steps, episode_data['cumulative_reward'], 'r-', alpha=0.7)
        axes[1, 1].set_title('Cumulative Reward')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            plt.savefig(os.path.join(save_dir, f'episode_{episode_idx + 1:03d}_lightweight.png'), 
                       dpi=150, bbox_inches='tight')  # Reduced DPI for speed
        except Exception as e:
            logger.error(f"Error saving lightweight plot for episode {episode_idx + 1}: {e}")
        finally:
            plt.close()
    
    def _create_summary_plots(self, results_dir: str) -> None:
        """Create summary plots across all episodes."""
        try:
            if self.episode_rewards:
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(self.episode_rewards)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Episode Rewards During Training')
                plt.grid(True, alpha=0.3)
                
                # Moving average
                if len(self.episode_rewards) > 10:
                    window = min(20, len(self.episode_rewards) // 5)  # Smaller window
                    moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', alpha=0.7, label=f'Moving Avg ({window})')
                    plt.legend()
                
                plt.subplot(1, 2, 2)
                if self.episode_phonemes:
                    phonemes_array = np.array(self.episode_phonemes)
                    for i in range(min(phonemes_array.shape[1], 3)):
                        plt.plot(phonemes_array[:, i], label=f'Agent {i}', alpha=0.7)
                    plt.xlabel('Episode')
                    plt.ylabel('Final Phoneme Count')
                    plt.title('Agent Phonemes per Episode')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'episode_metrics_lightweight.png'), dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating summary plots: {e}")
    
    def save_data(self, filepath: str) -> None:
        """Save callback data to file for later analysis."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_phonemes': self.episode_phonemes,
            'episode_gini': self.episode_gini,
            'episode_count': self.episode_count,
            'detailed_plotting_enabled': self.detailed_plotting
        }
        
        if self.detailed_plotting:
            data['episodes_step_data'] = self.episodes_step_data
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Lightweight callback data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving callback data: {e}")