import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import pickle

class CallbackPerEpisode(BaseCallback):
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
        
        # Episode-wise step data (for detailed episode plots)
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
            energy = info.get("energy", [0, 0, 0])
            
            # Store step data for current episode
            self.current_episode_data['rewards'].append(reward)
            self.current_episode_data['phonemes'].append(phonemes.copy())
            self.current_episode_data['gini'].append(current_gini)
            self.current_episode_data['actions'].append(action_number)
            self.current_episode_data['cumulative_reward'].append(self.current_episode_reward)
            self.current_episode_data['energy'].append(energy)
            
            # Log step-level metrics
            global_step = self.num_timesteps
            
            # Rewards
            self.writer.add_scalar('Step/Reward', reward, global_step)
            self.writer.add_scalar('Step/EnvReward', env_reward, global_step)
            
            # Phonemes per agent
            for i, phoneme_count in enumerate(phonemes):
                self.writer.add_scalar(f'Step_Phonemes/Agent_{i}_Phonemes', phoneme_count, global_step)
            
            self.writer.add_scalars(f'Step_Phonemes/All_Agent_Phonemes',{"Agent-1" : phonemes[0],"Agent-2" : phonemes[1],"Agent-3" : phonemes[2]},global_step)

            for i, energy_count in enumerate(energy):
                self.writer.add_scalar(f'Step_Energy/Agent_{i}_Energy', energy_count, global_step)
            
            self.writer.add_scalars(f'Step_Energy/All_Agent_Energy',{"Agent-1" : energy[0],"Agent-2" : energy[1],"Agent-3" : energy[2]},global_step)
            
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
                    self.writer.add_scalar(f'Step_Percentage/Agent_{i}_Percentage', percentage, global_step)
            
            # Check if episode ended
            dones = self.locals.get("dones", [False])
            if dones[0]:  # Episode ended
                self._log_episode_metrics(info)
                
        return True
    
    def _log_episode_metrics(self, final_info):
        """Log episode-level metrics."""
        self.episode_count += 1
        
        # Store episode data for detailed plotting
        self.episodes_step_data.append(self.current_episode_data.copy())
        
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
        self.current_episode_data = {
            'rewards': [],
            'phonemes': [],
            'gini': [],
            'actions': [],
            'cumulative_reward': [],
            'energy': []
        }
        
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
    
    def create_final_plots(self, results_dir: str, plot_episodes: list = None, max_episodes: int = 10):
        """
        Create final matplotlib plots for the paper, including detailed episode plots.
        
        Args:
            results_dir: Directory to save plots
            plot_episodes: List of specific episode numbers to plot (0-indexed). If None, plots first max_episodes
            max_episodes: Maximum number of episodes to plot if plot_episodes is None
        """
        os.makedirs(results_dir, exist_ok=True)
        
        # Create episode plots directory
        episodes_dir = os.path.join(results_dir, 'individual_episodes')
        os.makedirs(episodes_dir, exist_ok=True)
        
        # Determine which episodes to plot
        if plot_episodes is None:
            plot_episodes = list(range(min(len(self.episodes_step_data), max_episodes)))
        else:
            # Filter out invalid episode numbers
            plot_episodes = [ep for ep in plot_episodes if 0 <= ep < len(self.episodes_step_data)]
        
        # Create individual episode plots
        for episode_idx in plot_episodes:
            if episode_idx < len(self.episodes_step_data):
                self._create_episode_plot(episode_idx, episodes_dir)
        
        # Create summary plots
        self._create_summary_plots(results_dir)
        
        print(f"Created plots for {len(plot_episodes)} episodes in {episodes_dir}")
        print(f"Created summary plots in {results_dir}")
    
    def _create_episode_plot(self, episode_idx: int, save_dir: str):
        """Create a detailed plot for a single episode."""
        if episode_idx >= len(self.episodes_step_data):
            return
            
        episode_data = self.episodes_step_data[episode_idx]
        steps = range(len(episode_data['rewards']))
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f'Episode {episode_idx + 1} - Step-by-Step Analysis', fontsize=16)
        
        # Plot 1: Rewards over steps
        axes[0, 0].plot(steps, episode_data['rewards'], 'b-', alpha=0.7, label='Step Reward')
        axes[0, 0].plot(steps, episode_data['cumulative_reward'], 'r-', alpha=0.7, label='Cumulative Reward')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Rewards per Step')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Phonemes per agent over steps
        phonemes_array = np.array(episode_data['phonemes'])
        num_agents = phonemes_array.shape[1] if len(phonemes_array) > 0 else 3
        
        for agent_idx in range(num_agents):
            if len(phonemes_array) > 0:
                agent_phonemes = phonemes_array[:, agent_idx]
                axes[0, 1].plot(steps, agent_phonemes, label=f'Agent {agent_idx}', alpha=0.8)
        
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Phoneme Count')
        axes[0, 1].set_title('Phonemes per Agent')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gini coefficient over steps
        axes[1, 0].plot(steps, episode_data['gini'], 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gini Coefficient')
        axes[1, 0].set_title('Balance (Lower = More Equal)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Action distribution
        actions = episode_data['actions']
        unique_actions, action_counts = np.unique([a for a in actions if a >= 0], return_counts=True)
        
        if len(unique_actions) > 0:
            axes[1, 1].bar(unique_actions, action_counts, alpha=0.7)
            axes[1, 1].set_xlabel('Action')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Action Distribution')
            axes[1, 1].set_xticks(unique_actions)
        else:
            axes[1, 1].text(0.5, 0.5, 'No valid actions recorded', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Action Distribution')

        # Create arrays to track cumulative counts for each action
        # Graph 1 - wait and stop (actions 0, 1)
        wait_counts = np.cumsum(np.array(actions) == 0)
        stop_counts = np.cumsum(np.array(actions) == 1)

        # Graph 2 - stare_at actions (actions 2, 3, 4)
        stare_0_counts = np.cumsum(np.array(actions) == 2)
        stare_1_counts = np.cumsum(np.array(actions) == 3)
        stare_2_counts = np.cumsum(np.array(actions) == 4)

        # Graph 3 - encourage actions (actions 5, 6, 7)
        encourage_0_counts = np.cumsum(np.array(actions) == 5)
        encourage_1_counts = np.cumsum(np.array(actions) == 6)
        encourage_2_counts = np.cumsum(np.array(actions) == 7)

        axes[2, 0].plot(steps, wait_counts, label='Wait (0)', linewidth=2)
        axes[2, 0].plot(steps, stop_counts, label='Stop (1)', linewidth=2)
        axes[2, 0].set_title('Cumulative Wait and Stop Actions')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Cumulative Count')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # Set y-ticks to show all values for Graph 1
        max_val_g1 = max(wait_counts[-1], stop_counts[-1])
        if max_val_g1 < 20:  # Only show all ticks if not too many
            axes[2, 0].set_yticks(range(int(max_val_g1) + 1))

        # Graph 2 - stare_at actions at position [0, 1]
        axes[2, 1].plot(steps, stare_0_counts, label='Stare at 0 (2)', linewidth=2)
        axes[2, 1].plot(steps, stare_1_counts, label='Stare at 1 (3)', linewidth=2)
        axes[2, 1].plot(steps, stare_2_counts, label='Stare at 2 (4)', linewidth=2)
        axes[2, 1].set_title('Cumulative Stare At Actions')
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Cumulative Count')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # Set y-ticks to show all values for Graph 2
        max_val_g2 = max(stare_0_counts[-1], stare_1_counts[-1], stare_2_counts[-1])
        if max_val_g2 < 20:  # Only show all ticks if not too many
            axes[0, 1].set_yticks(range(int(max_val_g2) + 1))

        # Graph 3 - encourage actions at position [1, 0]
        axes[2, 2].plot(steps, encourage_0_counts, label='Encourage 0 (5)', linewidth=2)
        axes[2, 2].plot(steps, encourage_1_counts, label='Encourage 1 (6)', linewidth=2)
        axes[2, 2].plot(steps, encourage_2_counts, label='Encourage 2 (7)', linewidth=2)
        axes[2, 2].set_title('Cumulative Encourage Actions')
        axes[2, 2].set_xlabel('Step')
        axes[2, 2].set_ylabel('Cumulative Count')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'episode_{episode_idx + 1:03d}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_episode_plot_3(self, episode_idx: int, save_dir: str):
        """Create a detailed plot for a single episode."""
        print("--------------")
        if episode_idx >= len(self.episodes_step_data):
            return
            
        episode_data = self.episodes_step_data[episode_idx]
        print(episode_data)
        steps = range(len(episode_data['rewards']))
        
        # Create figure with subplots: 3 rows with different column counts
        fig = plt.figure(figsize=(18, 15))
        # Define a grid specification with 3 rows
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])
        
        # First row: 2 plots (spans 2 columns each)
        ax1 = fig.add_subplot(gs[0, 0:3])  # Rewards plot (spans first 2 columns)
        # ax2 = fig.add_subplot(gs[0, 2])    # Phonemes plot
        
        # Second row: 2 plots (spans 2 columns each)
        ax3 = fig.add_subplot(gs[1, 0:3])  # Gini plot (spans first 2 columns)
        # ax4 = fig.add_subplot(gs[1, 2])    # Action distribution plot
        
        # Third row: 3 action type plots
        ax5 = fig.add_subplot(gs[2, 0])    # Wait/Stop actions
        ax6 = fig.add_subplot(gs[2, 1])    # Stare actions
        ax7 = fig.add_subplot(gs[2, 2])    # Encourage actions
        ax8 = fig.add_subplot(gs[3, :])    # Energy
        
        fig.suptitle(f'Episode {episode_idx + 1} - Step-by-Step Analysis', fontsize=16)
        
        # Plot 1: Rewards over steps
        ax1.plot(steps, episode_data['rewards'], 'b-', alpha=0.7, label='Step Reward')
        # ax1.plot(steps, episode_data['cumulative_reward'], 'r-', alpha=0.7, label='Cumulative Reward')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Rewards per Step')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Phonemes per agent over steps
        phonemes_array = np.array(episode_data['phonemes'])
        num_agents = phonemes_array.shape[1] if len(phonemes_array) > 0 else 3

        colors = ['blue', 'orange', 'green']
        
        for agent_idx in range(num_agents):
            if len(phonemes_array) > 0:
                agent_phonemes = phonemes_array[:, agent_idx]
                ax3.plot(steps, agent_phonemes, label=f'Agent {agent_idx}', color=colors[agent_idx], alpha=1)
        
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Phoneme Count')
        ax3.set_title('Phonemes per Agent')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # Plot 4: Action distribution
        actions = episode_data['actions']
    
        wait_counts = np.cumsum(np.array(actions) == 0)
        
        # Graph 2 - stare_at actions (actions 1, 2, 3)
        stare_0_counts = np.cumsum(np.array(actions) == 1)
        stare_1_counts = np.cumsum(np.array(actions) == 2)
        stare_2_counts = np.cumsum(np.array(actions) == 3)
        
        # Graph 3 - encourage actions (actions 4, 5, 6)
        encourage_0_counts = np.cumsum(np.array(actions) == 4)
        encourage_1_counts = np.cumsum(np.array(actions) == 5)
        encourage_2_counts = np.cumsum(np.array(actions) == 6)
        
        # Wait/Stop actions plot
        ax5.plot(steps, wait_counts, color="#f47906", linewidth=2, label='Wait (0)')  # Dark Magenta
        ax5.set_title('Cumulative Wait Actions')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Count')
        ax5.legend()
        ax5.grid(True)
        
        # Stare actions plot
        ax6.plot(steps, stare_0_counts, 'b-', linewidth=2, label='Stare at 0 (2)')  
        ax6.plot(steps, stare_1_counts, color='orange', linewidth=2, label='Stare at 1 (3)')  
        ax6.plot(steps, stare_2_counts, 'g-', linewidth=2, label='Stare at 2 (4)')  
        ax6.set_title('Cumulative Stare At Actions')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Count')
        ax6.legend()
        ax6.grid(True)
        
        # Set y-ticks to show all values for Graph 2
        max_val_g2 = max(stare_0_counts[-1], stare_1_counts[-1], stare_2_counts[-1])
        if max_val_g2 < 20:  # Only show all ticks if not too many
            ax6.set_yticks(range(int(max_val_g2) + 1))
        
        # Encourage actions plot
        ax7.plot(steps, encourage_0_counts, 'b-', linewidth=2, label='Encourage 0 (5)')  
        ax7.plot(steps, encourage_1_counts, color='orange', linewidth=2, label='Encourage 1 (6)')  
        ax7.plot(steps, encourage_2_counts, 'g-', linewidth=2, label='Encourage 2 (7)')  
        ax7.set_title('Cumulative Encourage Actions')
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Count')
        ax7.legend()
        ax7.grid(True)

        ax8.plot(steps, episode_data['energy'], 'b-', alpha=0.7, label='Step Reward')
        # ax1.plot(steps, episode_data['cumulative_reward'], 'r-', alpha=0.7, label='Cumulative Reward')
        ax8.set_xlabel('Step')
        ax8.set_ylabel('energy')
        ax8.set_title('Rewards per Step')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Set y-ticks to show all values for Graph 3
        max_val_g3 = max(encourage_0_counts[-1], encourage_1_counts[-1], encourage_2_counts[-1])
        if max_val_g3 < 20:  # Only show all ticks if not too many
            ax7.set_yticks(range(int(max_val_g3) + 1))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
        plt.savefig(os.path.join(save_dir, f'episode_{episode_idx + 1:03d}.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_plots(self, results_dir: str):
        """Create summary plots across all episodes."""
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
    
    def save_data(self, filepath: str):
        """Save callback data to file for later analysis."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_phonemes': self.episode_phonemes,
            'episode_gini': self.episode_gini,
            'episode_actions': self.episode_actions,
            'episodes_step_data': self.episodes_step_data,
            'episode_count': self.episode_count
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Callback data saved to {filepath}")
    
    def load_data(self, filepath: str):
        """Load callback data from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.episode_rewards = data['episode_rewards']
        self.episode_phonemes = data['episode_phonemes']
        self.episode_gini = data['episode_gini']
        self.episode_actions = data['episode_actions']
        self.episodes_step_data = data['episodes_step_data']
        self.episode_count = data['episode_count']
        print(f"Callback data loaded from {filepath}")