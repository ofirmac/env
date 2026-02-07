import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import pickle
from typing import Dict, List, Optional, Union, Any
import numpy.typing as npt
from loguru import logger


class CallbackPerEpisode(BaseCallback):
    """
    Enhanced callback for multi-agent conversation environments.
    
    Tracks and logs detailed metrics including:
    - Agent phoneme counts and distributions
    - Energy levels per agent
    - Gini coefficient for conversation balance
    - Action statistics and patterns
    - Episode-level and step-level rewards
    
    Args:
        log_dir: Directory for TensorBoard logs
        max_stored_episodes: Maximum episodes to store for plotting (memory limit)
    """
    @logger.catch
    def __init__(self, log_dir: str = "./tensorboard_logs/", max_stored_episodes: int = 10_000) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        self.max_stored_episodes = max_stored_episodes
        
        # Episode-level tracking
        self.episode_rewards = []
        self.avg_episode_rewards = []
        self.episode_phonemes = []
        self.episode_gini = []
        self.episode_actions = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.episode_count = 0
        
        # Episode-wise step data (for detailed episode plots)
        self.episodes_step_data = []
        self.current_episode_data = {
            'rewards': [],           # PPO rewards (with shaping)
            'env_rewards': [],       # Base environment rewards
            'phonemes': [],
            'gini': [],
            'actions': [],
            'cumulative_reward': [],
            'energy': []
        }
    @logger.catch    
    def _on_training_start(self) -> None:
        """Initialize TensorBoard writer."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
    @logger.catch    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Get info from the environment
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0])
        
        if len(infos) > 0 and len(rewards) > 0:
            info = infos[0]
            logger.debug(f"{rewards=}")
            reward = rewards[0]
            logger.debug(f"{reward=}")
            
            # Track current episode
            self.current_episode_reward += reward
            self.current_episode_steps += 1
            
            # Get metrics from info with safe extraction
            phonemes = self._safe_get_metric(info, "phoneme", [0, 0, 0])
            gini_history = self._safe_get_metric(info, "gini_history", [])
            current_gini = self._safe_get_metric(info, "current_gini", 0)  # FIXED: Use direct gini value
            action_number = self._safe_get_metric(info, "action_number", -1)
            env_reward = self._safe_get_metric(info, "env_reward", 0)  # Base reward without shaping
            total_reward = self._safe_get_metric(info, "total_reward", reward)  # FIXED: Total reward with shaping
            energy = self._safe_get_metric(info, "energy", [0, 0, 0])
            
            # Store step data for current episode
            # FIXED: Store the actual reward used by PPO (includes shaping)
            self.current_episode_data['rewards'].append(reward)  # PPO reward (with shaping)
            self.current_episode_data['env_rewards'].append(env_reward)  # Base environment reward
            self.current_episode_data['phonemes'].append(phonemes.copy() if isinstance(phonemes, list) else list(phonemes))
            self.current_episode_data['gini'].append(current_gini)
            self.current_episode_data['actions'].append(action_number)
            self.current_episode_data['cumulative_reward'].append(self.current_episode_reward)
            self.current_episode_data['energy'].append(energy.copy() if isinstance(energy, list) else list(energy))
            
            # Log step-level metrics
            global_step = self.num_timesteps
            
            # FIXED: Log both PPO reward and environment reward for comparison
            self.writer.add_scalar('Step/PPO_Reward', reward, global_step)  # Total reward used by PPO
            self.writer.add_scalar('Step/Env_Reward', env_reward, global_step)  # Base environment reward
            if total_reward != reward:
                self.writer.add_scalar('Step/Total_Reward', total_reward, global_step)  # In case there's a difference
            
            # Dynamic agent handling for phonemes
            num_agents = len(phonemes) if isinstance(phonemes, (list, np.ndarray)) else 3
            for i in range(num_agents):
                if i < len(phonemes):
                    self.writer.add_scalar(f'Step_Phonemes/Agent_{i}_Phonemes', phonemes[i], global_step)
            
            # Dynamic TensorBoard logging for all agents
            if num_agents > 0 and len(phonemes) >= num_agents:
                agent_phoneme_dict = {f"Agent_{i}": phonemes[i] for i in range(num_agents)}
                self.writer.add_scalars('Step_Phonemes/All_Agent_Phonemes', agent_phoneme_dict, global_step)

            # Energy logging with dynamic agent handling
            num_energy_agents = len(energy) if isinstance(energy, (list, np.ndarray)) else 3
            for i in range(num_energy_agents):
                if i < len(energy):
                    self.writer.add_scalar(f'Step_Energy/Agent_{i}_Energy', energy[i], global_step)
            
            if num_energy_agents > 0 and len(energy) >= num_energy_agents:
                agent_energy_dict = {f"Agent_{i}": energy[i] for i in range(num_energy_agents)}
                self.writer.add_scalars('Step_Energy/All_Agent_Energy', agent_energy_dict, global_step)
            
            # Gini coefficient
            self.writer.add_scalar('Step/Gini_Coefficient', current_gini, global_step)
            
            # Actions
            if action_number >= 0:
                self.writer.add_scalar('Step/Action', action_number, global_step)
            
            # Phoneme distribution
            total_phonemes = sum(phonemes) if isinstance(phonemes, (list, np.ndarray)) else 0
            if total_phonemes > 0:
                for i in range(num_agents):
                    if i < len(phonemes):
                        percentage = (phonemes[i] / total_phonemes) * 100
                        self.writer.add_scalar(f'Step_Percentage/Agent_{i}_Percentage', percentage, global_step)
            
            # Check if episode ended
            dones = self.locals.get("dones", [False])
            if dones[0]:  # Episode ended
                self._log_episode_metrics(info)
                
        return True
    @logger.catch
    def _safe_get_metric(self, info: Dict[str, Any], key: str, default_value: Any) -> Any:
        """Safely extract metrics from info dict with validation."""
        try:
            value = info.get(key, default_value)
            if key in ["phoneme", "energy"] and not isinstance(value, (list, np.ndarray)):
                logger.warning(f"Expected list/array for {key}, got {type(value)}")
                return default_value
            return value
        except Exception as e:
            logger.warning(f"Error extracting {key}: {e}")
            return default_value
    @logger.catch
    def _get_num_agents(self) -> int:
        """Dynamically determine number of agents from data."""
        if self.episode_phonemes:
            return len(self.episode_phonemes[0])
        return 3  # Default fallback
    @logger.catch
    def _log_episode_metrics(self, final_info: Dict[str, Any]) -> None:
        """Log episode-level metrics."""
        self.episode_count += 1
        
        # Memory management: limit stored episodes
        if len(self.episodes_step_data) >= self.max_stored_episodes:
            self.episodes_step_data.pop(0)  # Remove oldest
        
        # Store episode data for detailed plotting
        self.episodes_step_data.append(self.current_episode_data.copy())
        
        # Episode reward
        self.episode_rewards.append(self.current_episode_reward)
        self.writer.add_scalar('Episode/Total_Reward', self.current_episode_reward, self.episode_count)
        
        # Avg reward
        avg_reward = self.current_episode_reward / self.current_episode_steps
        self.avg_episode_rewards.append(avg_reward)
        self.writer.add_scalar('Episode/Avg_Total_Reward', avg_reward, self.episode_count)
        
        # Avoid division by zero
        if self.current_episode_steps > 0:
            self.writer.add_scalar('Episode/Average_Reward', self.current_episode_reward / self.current_episode_steps, self.episode_count)
        
        # Final phoneme counts
        final_phonemes = self._safe_get_metric(final_info, "phoneme", [0, 0, 0])
        self.episode_phonemes.append(final_phonemes.copy() if isinstance(final_phonemes, list) else list(final_phonemes))
        
        num_agents = len(final_phonemes) if isinstance(final_phonemes, (list, np.ndarray)) else 3
        for i in range(num_agents):
            if i < len(final_phonemes):
                self.writer.add_scalar(f'Episode/Agent_{i}_Final_Phonemes', final_phonemes[i], self.episode_count)
        
        # Episode balance metrics
        total_phonemes = sum(final_phonemes) if isinstance(final_phonemes, (list, np.ndarray)) else 0
        if total_phonemes > 0:
            # Phoneme percentages
            percentages = [(count / total_phonemes) * 100 for count in final_phonemes]
            for i, percentage in enumerate(percentages):
                if i < num_agents:
                    self.writer.add_scalar(f'Episode/Agent_{i}_Final_Percentage', percentage, self.episode_count)
            
            # Balance metrics
            std_dev = np.std(final_phonemes)
            self.writer.add_scalar('Episode/Phoneme_StdDev', std_dev, self.episode_count)
            
            # Gini coefficient
            gini_history = self._safe_get_metric(final_info, "gini_history", [])
            if gini_history:
                final_gini = gini_history[-1]
                self.episode_gini.append(final_gini)
                self.writer.add_scalar('Episode/Final_Gini', final_gini, self.episode_count)
        
        # Action statistics
        action_stats = self._safe_get_metric(final_info, "actions_stats", [])
        if len(action_stats) > 0:
            for i, action_count in enumerate(action_stats):
                self.writer.add_scalar(f'Episode/Action_{i}_Count', action_count, self.episode_count)
        
        # Episode length
        self.writer.add_scalar('Episode/Length', self.current_episode_steps, self.episode_count)
        
        # Reset for next episode
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.current_episode_data = {
            'rewards': [],           # PPO rewards (with shaping)
            'env_rewards': [],       # FIXED: Base environment rewards
            'phonemes': [],
            'gini': [],
            'actions': [],
            'cumulative_reward': [],
            'energy': []
        }
        
        # Log episode summary every 10 episodes
        if self.episode_count % 10 == 0:
            self._log_summary_statistics()
    @logger.catch
    def _log_summary_statistics(self) -> None:
        """Log summary statistics over recent episodes."""
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            recent_phonemes = self.episode_phonemes[-10:]
            
            # Average metrics over last 10 episodes
            avg_reward = np.mean(recent_rewards)
            self.writer.add_scalar('Summary/Avg_Episode_Reward_10', avg_reward, self.episode_count)
            
            # Average phoneme distribution
            if recent_phonemes:
                avg_phonemes = np.mean(recent_phonemes, axis=0)
                for i, avg_count in enumerate(avg_phonemes):
                    self.writer.add_scalar(f'Summary/Avg_Agent_{i}_Phonemes_10', avg_count, self.episode_count)
                
                # Balance over last 10 episodes
                phoneme_std = np.std(avg_phonemes)
                self.writer.add_scalar('Summary/Avg_Balance_StdDev_10', phoneme_std, self.episode_count)
    @logger.catch
    def _on_training_end(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
    @logger.catch
    def create_final_plots(self, results_dir: str, plot_episodes: Optional[List[int]] = None, max_episodes: int = 10) -> None:
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
    @logger.catch
    def _create_episode_plot(self, episode_idx: int, save_dir: str) -> None:
        """Create a detailed plot for a single episode with improved layout and error handling."""
        if episode_idx >= len(self.episodes_step_data):
            logger.warning(f"Episode {episode_idx} not found in stored data")
            return
            
        episode_data = self.episodes_step_data[episode_idx]
        
        # Validate episode data
        if not episode_data or not episode_data.get('rewards'):
            logger.warning(f"No data available for episode {episode_idx}")
            return
            
        steps = range(len(episode_data['rewards']))
        
        # Create figure with improved grid layout
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 2])
        
        # Row 1: Rewards
        ax1 = fig.add_subplot(gs[0, :])
        
        # Row 2: Phonemes  
        ax2 = fig.add_subplot(gs[1, :])
        
        # Row 3: Action plots
        ax3 = fig.add_subplot(gs[2, 0])  # Wait actions
        ax4 = fig.add_subplot(gs[2, 1])  # Stare actions
        ax5 = fig.add_subplot(gs[2, 2])  # Encourage actions
        
        # Row 4: Energy
        # ax6 = fig.add_subplot(gs[3, :])
        
        fig.suptitle(f'Episode {episode_idx + 1} - Step-by-Step Analysis', fontsize=16)
        
        # Plot 1: Rewards over steps - FIXED to show both reward types
        ax1.plot(steps, episode_data['rewards'], 'b-', alpha=0.7, label='PPO Reward (with shaping)')
        if 'env_rewards' in episode_data and episode_data['env_rewards']:
            ax1.plot(steps, episode_data['env_rewards'], 'r-', alpha=0.7, label='Environment Reward (base)')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Rewards per Step')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Phonemes per agent over steps
        phonemes_array = np.array(episode_data['phonemes'])
        if len(phonemes_array) > 0 and phonemes_array.ndim > 1:
            num_agents = phonemes_array.shape[1]
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Support more agents
            
            for agent_idx in range(min(num_agents, len(colors))):
                agent_phonemes = phonemes_array[:, agent_idx]
                ax2.plot(steps, agent_phonemes, label=f'Agent {agent_idx}', 
                        color=colors[agent_idx % len(colors)], alpha=1)
        else:
            ax2.text(0.5, 0.5, 'No phoneme data available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Phoneme Count')
        ax2.set_title('Phonemes per Agent')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Action analysis with FIXED action mapping
        actions = episode_data['actions']
        
        # FIXED: Correct action mapping based on environment
        # Environment: 0=wait, 1=stare_at_0, 2=stare_at_1, 3=stare_at_2, 4=encourage_0, 5=encourage_1, 6=encourage_2
        wait_counts = np.cumsum(np.array(actions) == 0)
        
        stare_0_counts = np.cumsum(np.array(actions) == 1)  # FIXED: was 2
        stare_1_counts = np.cumsum(np.array(actions) == 2)  # FIXED: was 3
        stare_2_counts = np.cumsum(np.array(actions) == 3)  # FIXED: was 4
        
        encourage_0_counts = np.cumsum(np.array(actions) == 4)  # FIXED: was 5
        encourage_1_counts = np.cumsum(np.array(actions) == 5)  # FIXED: was 6
        encourage_2_counts = np.cumsum(np.array(actions) == 6)  # FIXED: was 7
        
        # Wait actions plot
        ax3.plot(steps, wait_counts, color="#f47906", linewidth=2, label='Wait (0)')
        ax3.set_title('Cumulative Wait Actions')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True)
        
        # Stare actions plot
        ax4.plot(steps, stare_0_counts, 'b-', linewidth=2, label='Stare at 0 (1)')  # FIXED label
        ax4.plot(steps, stare_1_counts, color='orange', linewidth=2, label='Stare at 1 (2)')  # FIXED label
        ax4.plot(steps, stare_2_counts, 'g-', linewidth=2, label='Stare at 2 (3)')  # FIXED label
        ax4.set_title('Cumulative Stare At Actions')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Count')
        ax4.legend()
        ax4.grid(True)
        
        # Set y-ticks for stare actions if reasonable number
        max_val_stare = max(stare_0_counts[-1] if len(stare_0_counts) > 0 else 0, 
                           stare_1_counts[-1] if len(stare_1_counts) > 0 else 0, 
                           stare_2_counts[-1] if len(stare_2_counts) > 0 else 0)
        if max_val_stare < 20:
            ax4.set_yticks(range(int(max_val_stare) + 1))
        
        # Encourage actions plot
        ax5.plot(steps, encourage_0_counts, 'b-', linewidth=2, label='Encourage 0 (4)')  # FIXED label
        ax5.plot(steps, encourage_1_counts, color='orange', linewidth=2, label='Encourage 1 (5)')  # FIXED label
        ax5.plot(steps, encourage_2_counts, 'g-', linewidth=2, label='Encourage 2 (6)')  # FIXED label
        ax5.set_title('Cumulative Encourage Actions')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Count')
        ax5.legend()
        ax5.grid(True)

        # --- Keep same y-axis scale for all action plots ---
        max_y = max(
            wait_counts[-1] if len(wait_counts) > 0 else 0,
            stare_0_counts[-1] if len(stare_0_counts) > 0 else 0,
            stare_1_counts[-1] if len(stare_1_counts) > 0 else 0,
            stare_2_counts[-1] if len(stare_2_counts) > 0 else 0,
            encourage_0_counts[-1] if len(encourage_0_counts) > 0 else 0,
            encourage_1_counts[-1] if len(encourage_1_counts) > 0 else 0,
            encourage_2_counts[-1] if len(encourage_2_counts) > 0 else 0
        )

        # Add a small margin
        ylim_max = max_y + max(1, int(max_y * 0.05))

        for ax in [ax3, ax4, ax5]:
            ax.set_ylim(0, ylim_max)

        
        # Set y-ticks for encourage actions if reasonable number
        max_val_encourage = max(encourage_0_counts[-1] if len(encourage_0_counts) > 0 else 0,
                               encourage_1_counts[-1] if len(encourage_1_counts) > 0 else 0,
                               encourage_2_counts[-1] if len(encourage_2_counts) > 0 else 0)
        if max_val_encourage < 20:
            ax5.set_yticks(range(int(max_val_encourage) + 1))
        
        # FIXED: Energy plot with proper array handling
        # TODO: Remove if it work 
        # energy_data = episode_data['energy']
        # if energy_data:
        #     try:
        #         energy_array = np.array(energy_data)
        #         if energy_array.ndim > 1 and energy_array.shape[1] > 0:
        #             colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        #             for i in range(min(energy_array.shape[1], len(colors))):
        #                 ax6.plot(steps, energy_array[:, i], 
        #                         label=f'Agent {i} Energy', 
        #                         color=colors[i % len(colors)], alpha=0.8)
        #         else:
        #             # Handle 1D energy data
        #             ax6.plot(steps, energy_array, 'b-', alpha=0.7, label='Energy')
        #     except Exception as e:
        #         logger.warning(f"Error plotting energy data: {e}")
        #         ax6.text(0.5, 0.5, 'Energy data format error', 
        #                 transform=ax6.transAxes, ha='center', va='center')
        # else:
        #     ax6.text(0.5, 0.5, 'No energy data available', 
        #             transform=ax6.transAxes, ha='center', va='center')
        
        # ax6.set_xlabel('Step')
        # ax6.set_ylabel('Energy')
        # ax6.set_title('Energy Levels per Agent')
        # ax6.legend()
        # ax6.grid(True, alpha=0.3)

                # --- Energy plots: split into individual graphs per agent on last row ---
                # --- Energy plots: one on top of the other (per agent) ---
        energy_data = episode_data.get("energy")

        if energy_data:
            try:
                energy_array = np.array(energy_data)  # shape: (steps, num_agents) or (steps,)

                # Handle 1D energy as a single-agent case
                if energy_array.ndim == 1:
                    energy_array = energy_array[:, None]  # (steps,) -> (steps, 1)

                num_agents = energy_array.shape[1]

                # Subdivide the last row (row 3) into num_agents rows, 1 column
                energy_gs = gs[3, :].subgridspec(num_agents, 1)

                colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

                # Shared y-limits across all energy plots (nice for comparison)
                max_energy = float(np.max(energy_array))
                min_energy = float(np.min(energy_array))
                y_margin = max(1e-3, (max_energy - min_energy) * 0.05)
                y_min = min_energy - y_margin
                y_max = max_energy + y_margin

                energy_axes = []

                for agent_idx in range(num_agents):
                    # First axis normal, others share x-axis to align steps
                    if agent_idx == 0:
                        ax_energy = fig.add_subplot(energy_gs[agent_idx, 0])
                    else:
                        ax_energy = fig.add_subplot(
                            energy_gs[agent_idx, 0],
                            sharex=energy_axes[0]
                        )

                    energy_axes.append(ax_energy)

                    ax_energy.plot(
                        steps,
                        energy_array[:, agent_idx],
                        color=colors[agent_idx % len(colors)],
                        alpha=0.9,
                        label=f"Agent {agent_idx} Energy",
                    )
                    ax_energy.set_ylabel("Energy")
                    ax_energy.set_title(f"Energy – Agent {agent_idx}")
                    ax_energy.grid(True, alpha=0.3)
                    ax_energy.set_ylim(y_min, y_max)
                    ax_energy.legend()

                # Only show x-label on the last energy plot
                energy_axes[-1].set_xlabel("Step")

            except Exception as e:
                logger.warning(f"Error plotting energy data: {e}")
                ax_energy_fallback = fig.add_subplot(gs[3, :])
                ax_energy_fallback.text(
                    0.5, 0.5, "Energy data format error",
                    transform=ax_energy_fallback.transAxes,
                    ha="center", va="center"
                )
                ax_energy_fallback.set_axis_off()
        else:
            ax_energy_empty = fig.add_subplot(gs[3, :])
            ax_energy_empty.text(
                0.5, 0.5, "No energy data available",
                transform=ax_energy_empty.transAxes,
                ha="center", va="center"
            )
            ax_energy_empty.set_axis_off()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
        
        # Save with error handling
        try:
            plt.savefig(os.path.join(save_dir, f'episode_{episode_idx + 1:03d}.png'), 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot for episode {episode_idx + 1}")
        except Exception as e:
            logger.error(f"Error saving plot for episode {episode_idx + 1}: {e}")
        finally:
            plt.close()
    @logger.catch
    def _create_summary_plots(self, results_dir: str) -> None:
        """Create summary plots across all episodes with improved error handling."""
        try:
            # Episode rewards plot
            if self.avg_episode_rewards:
                logger.info(f"{self.avg_episode_rewards=}")
                plt.figure(figsize=(18, 6))
                
                plt.subplot(1, 2, 1)
                plt.plot(self.avg_episode_rewards)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Episode Rewards During Training')
                plt.grid(True, alpha=0.3)
                
                # Moving average
                if len(self.avg_episode_rewards) > 10:
                    window = min(50, len(self.avg_episode_rewards) // 10)
                    moving_avg = np.convolve(self.avg_episode_rewards, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(self.avg_episode_rewards)), moving_avg, 'r-', alpha=0.7, label=f'Moving Avg ({window})')
                    plt.legend()
                
                plt.subplot(1, 2, 2)
                if self.episode_phonemes:
                    logger.info(f"{self.episode_phonemes=}")
                    avg = [[x / 10 for x in sublist] for sublist in self.episode_phonemes]
                    phonemes_array = np.array(avg)
                    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
                    for i in range(min(phonemes_array.shape[1], len(colors))):
                        plt.plot(phonemes_array[:, i], label=f'Agent {i}', 
                                color=colors[i % len(colors)], alpha=0.7)
                    plt.xlabel('Episode')
                    plt.ylabel('Final Phoneme Count')
                    plt.title('Agent Avg Phonemes per Episode')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'episode_metrics.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Balance analysis
            if self.episode_phonemes:
                plt.figure(figsize=(18,8))
                
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
                
        except Exception as e:
            logger.error(f"Error creating summary plots: {e}")
    
    @logger.catch
    def plot_cumulative_actions_single_figure(
        self,
        results_dir: str,
        filename: str = "cumulative_actions_all.png",
    ) -> None:
        """
        Plot cumulative actions across ALL episodes in ONE figure split into 3 rows:
            1. Wait
            2. Stare at
            3. Encourage
        """
        os.makedirs(results_dir, exist_ok=True)

        if not self.episodes_step_data:
            logger.warning("No episodes_step_data available – nothing to plot.")
            return

        # --- Flatten all actions across all episodes ---
        all_actions = []
        for ep in self.episodes_step_data:
            if "actions" in ep:
                all_actions.extend([a for a in ep["actions"] if isinstance(a, (int, np.integer)) and a >= 0])

        if not all_actions:
            logger.warning("No valid actions found – nothing to plot.")
            return

        all_actions = np.array(all_actions)
        steps = np.arange(len(all_actions))

        # --- Create figure with 3 stacked plots ---
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        fig.suptitle("Cumulative Action Usage Across Training", fontsize=18)

        # ========== 1) WAIT ==========
        wait_mask = (all_actions == 0).astype(int)
        wait_cum = np.cumsum(wait_mask)
        axes[0].plot(steps, wait_cum, linewidth=2, color="#f47906", label="Wait (0)")
        axes[0].set_ylabel("Count"); axes[0].set_title("Wait")
        axes[0].grid(True, alpha=0.3); axes[0].legend()

        # ========== 2) STARE ==========
        stare_colors = ["blue", "orange", "green"]
        stare_labels = ["Stare at 0 (1)", "Stare at 1 (2)", "Stare at 2 (3)"]
        for action_id, color, label in zip([1, 2, 3], stare_colors, stare_labels):
            mask = (all_actions == action_id).astype(int)
            axes[1].plot(steps, np.cumsum(mask), linewidth=2, color=color, label=label)
        axes[1].set_ylabel("Count"); axes[1].set_title("Stare")
        axes[1].grid(True, alpha=0.3); axes[1].legend()

        # ========== 3) ENCOURAGE ==========
        encour_colors = ["blue", "orange", "green"]
        encour_labels = ["Encourage 0 (4)", "Encourage 1 (5)", "Encourage 2 (6)"]
        for action_id, color, label in zip([4, 5, 6], encour_colors, encour_labels):
            mask = (all_actions == action_id).astype(int)
            axes[2].plot(steps, np.cumsum(mask), linewidth=2, color=color, label=label)
        axes[2].set_ylabel("Count"); axes[2].set_title("Encourage")
        axes[2].set_xlabel("Global Step Across Training")
        axes[2].grid(True, alpha=0.3); axes[2].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(results_dir, filename)

        try:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            logger.info(f"Cumulative action (single figure) plot saved to {out_path}")
        except Exception as e:
            logger.error(f"Error saving cumulative action figure: {e}")
        finally:
            plt.close()

    @logger.catch
    def plot_last_action_values_per_episode(
        self,
        results_dir: str,
        filename: str = "episode_last_action_values.png",
    ) -> None:
        """
        Plot only the LAST cumulative action count from each episode.
        Each episode contributes one number per action:
            - Final count of Wait
            - Final counts of Stare at 0/1/2
            - Final counts of Encourage 0/1/2

        Figure layout:
            Row 1 -> Wait
            Row 2 -> Stare
            Row 3 -> Encourage
        """
        os.makedirs(results_dir, exist_ok=True)

        if not self.episodes_step_data:
            logger.warning("No episodes_step_data available – nothing to plot.")
            return

        # Storage
        wait_vals = []
        stare_vals = {1: [], 2: [], 3: []}
        encourage_vals = {4: [], 5: [], 6: []}

        # --- collect the last cumulative value from every episode ---
        for ep in self.episodes_step_data:
            actions = ep.get("actions", [])
            if not actions:
                continue

            actions = np.array(actions)

            # Wait final
            wait_vals.append(np.sum(actions == 0))

            # Stare finals
            stare_vals[1].append(np.sum(actions == 1))
            stare_vals[2].append(np.sum(actions == 2))
            stare_vals[3].append(np.sum(actions == 3))

            # Encourage finals
            encourage_vals[4].append(np.sum(actions == 4))
            encourage_vals[5].append(np.sum(actions == 5))
            encourage_vals[6].append(np.sum(actions == 6))

        episodes = np.arange(len(wait_vals)) + 1  # 1-based indexing on X-axis

        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        fig.suptitle("Total Action Counts per Episode (final value only)", fontsize=18)

        # ========== 1) WAIT ==========
        axes[0].plot(episodes, wait_vals, linewidth=2, color="#f47906", label="Wait (0)")
        axes[0].set_ylabel("Count"); axes[0].set_title("Wait")
        axes[0].grid(True, alpha=0.3); axes[0].legend()

        # ========== 2) STARE ==========
        colors = ["blue", "orange", "green"]
        labels = ["Stare at 0 (1)", "Stare at 1 (2)", "Stare at 2 (3)"]
        for action_id, color, label in zip([1, 2, 3], colors, labels):
            axes[2 - 2 + 1]  # no avoid confusion
        for (action_id, color, label) in zip([1, 2, 3], colors, labels):
            axes[1].plot(episodes, stare_vals[action_id], color=color, linewidth=2, label=label)
        axes[1].set_ylabel("Count"); axes[1].set_title("Stare")
        axes[1].grid(True, alpha=0.3); axes[1].legend()

        # ========== 3) ENCOURAGE ==========
        colors = ["blue", "orange", "green"]
        labels = ["Encourage 0 (4)", "Encourage 1 (5)", "Encourage 2 (6)"]
        for (action_id, color, label) in zip([4, 5, 6], colors, labels):
            axes[2].plot(episodes, encourage_vals[action_id], color=color, linewidth=2, label=label)
        axes[2].set_ylabel("Count"); axes[2].set_title("Encourage")
        axes[2].set_xlabel("Episode")
        axes[2].grid(True, alpha=0.3); axes[2].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(results_dir, filename)

        try:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            logger.info(f"Episode last-action plot saved to {out_path}")
        except Exception as e:
            logger.error(f"Error saving last-action plot: {e}")
        finally:
            plt.close()


    @logger.catch
    def save_data(self, filepath: str) -> None:
        """Save callback data to file for later analysis."""
        data = {
            'episode_rewards': self.episode_rewards,
            'avg_episode_rewards': self.avg_episode_rewards,
            'episode_phonemes': self.episode_phonemes,
            'episode_gini': self.episode_gini,
            'episode_actions': self.episode_actions,
            'episodes_step_data': self.episodes_step_data,
            'episode_count': self.episode_count
        }
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Callback data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving callback data: {e}")
    @logger.catch
    def load_data(self, filepath: str) -> None:
        """Load callback data from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.episode_rewards = data.get('episode_rewards', [])
            self.avg_episode_rewards = data.get('avg_episode_rewards', [])
            self.episode_phonemes = data.get('episode_phonemes', [])
            self.episode_gini = data.get('episode_gini', [])
            self.episode_actions = data.get('episode_actions', [])
            self.episodes_step_data = data.get('episodes_step_data', [])
            self.episode_count = data.get('episode_count', 0)
            logger.info(f"Callback data loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading callback data: {e}")
        return data
