# Callback in Action: Real Example Walkthrough

## 🎬 **Real Training Session Example**

Let's walk through **exactly** what happens during a real training session with your callback.

### **Scenario: 3-Agent Conversation Environment**
- **Agent 0**: Shy person (speaks little)
- **Agent 1**: Balanced person (speaks moderately) 
- **Agent 2**: Talkative person (speaks a lot)
- **Goal**: Learn to balance the conversation (equal speaking time)

---

## 📋 **Episode 1: Step-by-Step Breakdown**

### **🏁 Training Starts**
```python
# This happens once at the beginning
callback._on_training_start()
```
**Console Output:**
```
2025-10-07 19:31:16 | INFO | TensorBoard writer initialized at ./tensorboard_logs/
```

### **🎯 Step 1: Agent Takes First Action**

**Environment State:**
- All agents silent: `phonemes = [0, 0, 0]`
- Perfect balance: `gini = 0.0` (no one spoke yet)
- Agent energy: `energy = [0.8, 0.3, 0.9]`

**Agent Decision:** Action 4 (encourage agent 1)

**Environment Response:**
```python
obs, reward, done, info = env.step(4)
# reward = 1.0 (good balance maintained)
# done = False (episode continues)
# info = {
#     'phoneme': [0, 1, 0],        # Agent 1 spoke 1 word
#     'current_gini': 0.0,         # Still balanced
#     'env_reward': 1.0,           # Base reward
#     'total_reward': 1.2,         # With +0.2 new speaker bonus
#     'action_number': 4,          # Encourage action
#     'energy': [0.8, 0.5, 0.9],  # Agent 1 energy increased
#     'num_of_step_env': 1
# }
```

**Callback Execution:**
```python
def _on_step(self) -> bool:
    # 1. Extract data safely
    phonemes = [0, 1, 0]          # From info
    reward = 1.2                  # PPO reward (with shaping)
    env_reward = 1.0              # Base environment reward
    current_gini = 0.0            # Balance score
    
    # 2. Update episode tracking
    self.current_episode_reward += 1.2  # Now = 1.2
    self.current_episode_steps += 1     # Now = 1
    
    # 3. Store step data
    self.current_episode_data['rewards'].append(1.2)        # PPO reward
    self.current_episode_data['env_rewards'].append(1.0)    # Base reward
    self.current_episode_data['phonemes'].append([0, 1, 0]) # Speech counts
    self.current_episode_data['gini'].append(0.0)           # Balance
    self.current_episode_data['actions'].append(4)          # Action taken
    
    # 4. Log to TensorBoard (creates real-time graphs)
    self.writer.add_scalar('Step/PPO_Reward', 1.2, global_step=1)
    self.writer.add_scalar('Step/Env_Reward', 1.0, global_step=1)
    self.writer.add_scalar('Step_Phonemes/Agent_0_Phonemes', 0, global_step=1)
    self.writer.add_scalar('Step_Phonemes/Agent_1_Phonemes', 1, global_step=1)
    self.writer.add_scalar('Step_Phonemes/Agent_2_Phonemes', 0, global_step=1)
    self.writer.add_scalar('Step/Gini_Coefficient', 0.0, global_step=1)
    
    # 5. Check if episode ended
    # done = False, so continue
    return True
```

### **🎯 Step 2: Agent Takes Second Action**

**Environment State:**
- Speech counts: `phonemes = [0, 1, 0]` (Agent 1 spoke 1 word)
- Still balanced: `gini = 0.0`
- Energies: `energy = [0.8, 0.5, 0.9]`

**Agent Decision:** Action 6 (encourage agent 2)

**Environment Response:**
```python
# Agent 2 speaks 2 words
# reward = 0.8 (balance getting worse)
# info = {
#     'phoneme': [0, 1, 2],        # Agent 2 now spoke 2 words
#     'current_gini': 0.33,        # Balance getting worse
#     'env_reward': 0.67,          # Base reward (1.0 - 0.33)
#     'total_reward': 0.87,        # With +0.2 new speaker bonus
#     'action_number': 6,
#     'energy': [0.8, 0.5, 0.7],  # Agent 2 energy decreased after speaking
# }
```

**Callback Execution:**
```python
# Step data now contains:
self.current_episode_data = {
    'rewards': [1.2, 0.87],           # PPO rewards
    'env_rewards': [1.0, 0.67],       # Base rewards  
    'phonemes': [[0,1,0], [0,1,2]],   # Speech progression
    'gini': [0.0, 0.33],              # Balance getting worse
    'actions': [4, 6],                # Encourage actions
    'cumulative_reward': [1.2, 2.07], # Running total
}

# Current episode totals:
self.current_episode_reward = 2.07
self.current_episode_steps = 2
```

### **🎯 Step 15: Episode Ends**

After 15 steps, conversation ends:

**Final Environment State:**
```python
# Final state after 15 steps
# info = {
#     'phoneme': [2, 8, 12],       # Final speech counts
#     'current_gini': 0.45,        # Unbalanced conversation
#     'env_reward': 0.55,          # Poor balance = low reward
#     'gini_history': [0.0, 0.33, 0.45, ...], # Balance over time
#     'actions_stats': [3, 2, 2, 4, 2, 2, 0], # Action counts
# }
# done = True  # Episode finished
```

**Episode End Callback:**
```python
def _log_episode_metrics(self, final_info):
    self.episode_count += 1  # Now = 1
    
    # 1. Store episode summary
    self.episode_rewards.append(12.5)      # Total episode reward
    self.episode_phonemes.append([2,8,12]) # Final speech counts
    self.episode_gini.append(0.45)         # Final balance score
    
    # 2. Log episode metrics to TensorBoard
    self.writer.add_scalar('Episode/Total_Reward', 12.5, episode=1)
    self.writer.add_scalar('Episode/Agent_0_Final_Phonemes', 2, episode=1)
    self.writer.add_scalar('Episode/Agent_1_Final_Phonemes', 8, episode=1)
    self.writer.add_scalar('Episode/Agent_2_Final_Phonemes', 12, episode=1)
    self.writer.add_scalar('Episode/Final_Gini', 0.45, episode=1)
    self.writer.add_scalar('Episode/Length', 15, episode=1)
    
    # 3. Calculate balance metrics
    total_phonemes = 2 + 8 + 12 = 22
    percentages = [9.1%, 36.4%, 54.5%]  # Very unbalanced!
    std_dev = 5.03  # High variability = bad balance
    
    # 4. Store detailed episode data for plotting
    self.episodes_step_data.append(self.current_episode_data.copy())
    
    # 5. Reset for next episode
    self.current_episode_reward = 0
    self.current_episode_steps = 0
    self.current_episode_data = {'rewards': [], 'phonemes': [], ...}
```

---

## 📊 **After 100 Episodes: What Data Do We Have?**

### **Episode-Level Data:**
```python
callback.episode_rewards = [12.5, 15.2, 18.7, ..., 45.8]     # 100 values
callback.episode_phonemes = [[2,8,12], [5,7,8], ..., [7,7,8]] # 100 arrays
callback.episode_gini = [0.45, 0.23, 0.15, ..., 0.05]        # Getting better!
```

### **Detailed Step Data (Last 50 episodes stored):**
```python
callback.episodes_step_data = [
    {  # Episode 51 data
        'rewards': [1.2, 0.87, 1.1, ..., 0.95],      # 15 step rewards
        'phonemes': [[0,1,0], [0,1,2], ..., [2,8,12]], # 15 step phoneme states
        'gini': [0.0, 0.33, 0.28, ..., 0.45],         # 15 balance scores
        'actions': [4, 6, 1, ..., 0],                  # 15 actions taken
    },
    # ... Episodes 52-100
]
```

---

## 🎨 **Creating Plots: What You Get**

### **1. Individual Episode Plot (Episode 100):**
```python
callback.create_final_plots(results_dir="./results", plot_episodes=[99])
```

**Generated Plot Shows:**
- **Top Panel**: Reward per step (Blue=PPO reward, Red=Environment reward)
- **Second Panel**: Speech counts per agent over steps
- **Third Panel**: Action usage (Wait, Stare At, Encourage actions)
- **Bottom Panel**: Energy levels per agent

### **2. Summary Plots:**
```python
callback.create_final_plots(results_dir="./results")
```

**Generated Plots Show:**
- **Episode Rewards**: Trend from episode 1 to 100 (should go up!)
- **Agent Phonemes**: How speech distribution evolved
- **Balance Analysis**: Gini coefficient over time (should go down!)

---

## 🔍 **Real TensorBoard Output**

While training, you can open TensorBoard and see:

### **Step-Level Graphs:**
```
Step/PPO_Reward: Live graph showing reward per step
Step/Env_Reward: Base environment reward per step
Step_Phonemes/Agent_0_Phonemes: Agent 0 speech over time
Step_Phonemes/Agent_1_Phonemes: Agent 1 speech over time  
Step_Phonemes/Agent_2_Phonemes: Agent 2 speech over time
Step/Gini_Coefficient: Balance score over time
```

### **Episode-Level Graphs:**
```
Episode/Total_Reward: Total reward per episode (trending up?)
Episode/Agent_0_Final_Phonemes: Final speech count for Agent 0
Episode/Final_Gini: Final balance score (trending down?)
Episode/Length: How long each episode lasted
```

### **Summary Graphs:**
```
Summary/Avg_Episode_Reward_10: Average reward over last 10 episodes
Summary/Avg_Agent_0_Phonemes_10: Average Agent 0 speech over last 10 episodes
Summary/Avg_Balance_StdDev_10: Average balance over last 10 episodes
```

---

## 🚀 **What This Tells You**

### **Training is Working If:**
- ✅ Episode rewards trend upward
- ✅ Gini coefficient trends downward (better balance)
- ✅ Speech distribution becomes more equal [7,7,8] vs [2,8,12]
- ✅ Agent learns to use different actions strategically

### **Training Has Problems If:**
- ❌ Rewards stay flat or decrease
- ❌ Gini coefficient stays high (poor balance)
- ❌ One agent dominates conversation consistently
- ❌ Agent only uses one type of action

### **Debug Using Callback Data:**
```python
# Check if agent is learning
if callback.episode_rewards[-10:] == callback.episode_rewards[-20:-10]:
    print("⚠️ Agent not improving - check hyperparameters")

# Check balance learning
recent_gini = np.mean(callback.episode_gini[-10:])
if recent_gini > 0.3:
    print("⚠️ Agent not learning to balance conversation")

# Check action diversity
last_episode_actions = callback.episodes_step_data[-1]['actions']
unique_actions = len(set(last_episode_actions))
if unique_actions < 3:
    print("⚠️ Agent using limited action set")
```

This is exactly how your callback works in practice - it's your window into understanding what your AI agent is learning and how well it's performing!