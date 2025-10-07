# Understanding Reinforcement Learning Callbacks: A Complete Guide

## 🎯 **What is a Callback in Reinforcement Learning?**

Think of a **callback** like a "spy" or "observer" that watches your AI agent while it's learning. 

### **Simple Analogy:**
Imagine you're teaching a child to ride a bicycle:
- **The child** = Your AI agent (PPO model)
- **The bicycle** = Your environment (conversation simulation)
- **You watching and taking notes** = The callback

While the child practices, you're constantly:
- ✅ Recording how many times they fall
- ✅ Measuring how far they go each attempt
- ✅ Noting what mistakes they make
- ✅ Tracking their improvement over time

**That's exactly what a callback does for AI training!**

---

## 🏗️ **Architecture Overview**

```mermaid
graph TD
    A[PPO Agent] --> B[Environment]
    B --> C[Step Result: obs, reward, done, info]
    C --> D[Callback._on_step()]
    D --> E[Extract Metrics]
    E --> F[Log to TensorBoard]
    E --> G[Store Episode Data]
    E --> H[Create Plots]
    
    B --> I[Episode Ends]
    I --> J[Callback._log_episode_metrics()]
    J --> K[Save Episode Summary]
    J --> L[Reset for Next Episode]
```

---

## 📚 **Step-by-Step Breakdown**

### **Step 1: Initialization (`__init__`)**

```python
def __init__(self, log_dir: str = "./tensorboard_logs/", max_stored_episodes: int = 100):
```

**What happens here:**
- 🏠 **Sets up storage locations** for logs and data
- 📊 **Creates empty lists** to store metrics:
  - `episode_rewards = []` - Total reward per episode
  - `episode_phonemes = []` - Speech counts per agent per episode
  - `episode_gini = []` - Conversation balance scores
- 🧠 **Initializes tracking variables**:
  - `current_episode_reward = 0` - Running total for current episode
  - `episode_count = 0` - How many episodes completed

**Think of it as:** Setting up your notebook and pen before watching the child practice bicycle riding.

---

### **Step 2: Training Starts (`_on_training_start`)**

```python
def _on_training_start(self) -> None:
    os.makedirs(self.log_dir, exist_ok=True)
    self.writer = SummaryWriter(self.log_dir)
```

**What happens here:**
- 📁 **Creates directories** for storing logs
- 📝 **Opens TensorBoard writer** (like opening a digital notebook)

**Think of it as:** Getting your camera ready before the child starts practicing.

---

### **Step 3: Every Single Step (`_on_step`) - THE HEART OF THE CALLBACK**

This is called **after every single action** the agent takes in the environment!

```python
def _on_step(self) -> bool:
    # Get info from the environment
    infos = self.locals.get("infos", [{}])
    rewards = self.locals.get("rewards", [0])
```

#### **3.1: Data Collection**
```python
info = infos[0]           # Environment information
reward = rewards[0]       # Reward agent received
```

**What's in `info`?** (From your conversation environment)
- `phoneme`: [0, 5, 3] - How many words each agent spoke
- `gini_history`: [0.2, 0.15, 0.1] - Balance scores over time
- `action_number`: 2 - What action was taken (0=wait, 1=stare_at_agent_0, etc.)
- `energy`: [0.8, 0.3, 0.9] - Energy levels of each agent

#### **3.2: Safe Data Extraction**
```python
phonemes = self._safe_get_metric(info, "phoneme", [0, 0, 0])
current_gini = self._safe_get_metric(info, "current_gini", 0)
```

**Why "safe"?** Sometimes the environment might not provide all data, so we use defaults to prevent crashes.

#### **3.3: Store Step Data**
```python
self.current_episode_data['rewards'].append(reward)
self.current_episode_data['phonemes'].append(phonemes)
self.current_episode_data['gini'].append(current_gini)
```

**Think of it as:** Writing down what happened in this exact moment:
- "Step 47: Agent got +0.8 reward, Agent 0 spoke 2 words, balance score is 0.15"

#### **3.4: Real-time Logging**
```python
self.writer.add_scalar('Step/PPO_Reward', reward, global_step)
self.writer.add_scalar('Step/Gini_Coefficient', current_gini, global_step)
```

**This creates live graphs** you can watch in TensorBoard while training!

#### **3.5: Episode End Detection**
```python
dones = self.locals.get("dones", [False])
if dones[0]:  # Episode ended
    self._log_episode_metrics(info)
```

**When episode ends:** Call special function to summarize everything that happened.

---

### **Step 4: Episode Ends (`_log_episode_metrics`)**

When an episode finishes (conversation ends), this function runs:

#### **4.1: Save Episode Summary**
```python
self.episode_rewards.append(self.current_episode_reward)
self.episode_phonemes.append(final_phonemes)
```

**Think of it as:** Writing a summary: "Episode 1: Total reward +15.6, Agent 0 spoke 12 words, Agent 1 spoke 8 words, Agent 2 spoke 15 words"

#### **4.2: Calculate Episode Statistics**
```python
# Balance metrics
std_dev = np.std(final_phonemes)  # How unequal was the conversation?
total_phonemes = sum(final_phonemes)  # Total words spoken
```

#### **4.3: Memory Management**
```python
if len(self.episodes_step_data) >= self.max_stored_episodes:
    self.episodes_step_data.pop(0)  # Remove oldest episode
```

**Why?** Prevents memory from growing infinitely during long training.

#### **4.4: Reset for Next Episode**
```python
self.current_episode_reward = 0
self.current_episode_steps = 0
self.current_episode_data = {'rewards': [], 'phonemes': [], ...}
```

**Think of it as:** Turning to a fresh page in your notebook for the next practice session.

---

### **Step 5: Training Ends (`_on_training_end`)**

```python
def _on_training_end(self) -> None:
    if self.writer:
        self.writer.close()
```

**Think of it as:** Closing your notebook and putting away your pen when practice is over.

---

## 📊 **What Data Gets Collected?**

### **Real-time (Every Step):**
- 🎯 **Rewards**: What reward did the agent get?
- 🗣️ **Speech**: How many words did each agent speak?
- ⚖️ **Balance**: How equal is the conversation?
- 🎬 **Actions**: What did the agent decide to do?
- ⚡ **Energy**: How active is each agent?

### **Episode Summary:**
- 📈 **Total Episode Reward**: Sum of all step rewards
- 🏁 **Final Speech Counts**: Total words per agent
- 📊 **Balance Metrics**: How equal was the final conversation?
- 📏 **Episode Length**: How many steps did it take?

---

## 🎨 **Visualization & Plotting**

### **TensorBoard (Real-time)**
While training runs, you can open TensorBoard and see:
- 📈 Live reward graphs
- 📊 Speech distribution charts
- ⚖️ Balance metrics over time

### **Matplotlib Plots (After training)**
```python
callback.create_final_plots(results_dir="./plots")
```

Creates detailed plots:
- **Individual Episode Plots**: Step-by-step breakdown of specific episodes
- **Summary Plots**: Trends across all episodes
- **Balance Analysis**: How conversation equality evolved

---

## 🔧 **Key Features of This Callback**

### **1. Dynamic Agent Support**
```python
num_agents = len(phonemes)  # Works with any number of agents
for i in range(num_agents):
    self.writer.add_scalar(f'Agent_{i}_Phonemes', phonemes[i], step)
```

**Not hardcoded to 3 agents** - works with 2, 4, 5, or any number!

### **2. Error Handling**
```python
def _safe_get_metric(self, info: Dict, key: str, default_value: Any):
    try:
        return info.get(key, default_value)
    except Exception as e:
        logger.warning(f"Error extracting {key}: {e}")
        return default_value
```

**Won't crash** if environment doesn't provide expected data.

### **3. Memory Management**
```python
if len(self.episodes_step_data) >= self.max_stored_episodes:
    self.episodes_step_data.pop(0)  # Remove oldest
```

**Prevents memory leaks** during long training sessions.

### **4. Dual Reward Tracking**
```python
self.current_episode_data['rewards'].append(reward)          # PPO reward (with shaping)
self.current_episode_data['env_rewards'].append(env_reward)  # Base environment reward
```

**Tracks both** the raw environment reward and the shaped reward used by PPO.

---

## 🚀 **How to Use It**

### **Basic Usage:**
```python
from src.callback.guest_callback_per_episode import CallbackPerEpisode

# Create callback
callback = CallbackPerEpisode(log_dir="./logs")

# Use with PPO training
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000, callback=callback)

# Create plots after training
callback.create_final_plots(results_dir="./results")
```

### **Advanced Usage:**
```python
# Limit memory usage
callback = CallbackPerEpisode(
    log_dir="./logs", 
    max_stored_episodes=50  # Only keep last 50 episodes
)

# Plot specific episodes
callback.create_final_plots(
    results_dir="./results",
    plot_episodes=[0, 10, 25, 49],  # Plot episodes 1, 11, 26, 50
    max_episodes=10
)
```

---

## 🎯 **Why Is This Useful?**

### **For Debugging:**
- 🐛 **See exactly when things go wrong**: "Reward dropped at step 23 of episode 15"
- 🔍 **Understand agent behavior**: "Agent always chooses 'wait' action"
- ⚖️ **Monitor balance**: "Conversations becoming more unequal over time"

### **For Research:**
- 📊 **Generate publication-quality plots**
- 📈 **Track training progress**
- 🧪 **Compare different training runs**
- 📝 **Document experimental results**

### **For Optimization:**
- 🚀 **Identify performance bottlenecks**
- 🎯 **Fine-tune hyperparameters**
- 🔧 **Adjust reward shaping**

---

## 🔄 **The Complete Flow**

```
1. Training Starts
   ↓
2. Agent takes action in environment
   ↓
3. Environment returns: observation, reward, done, info
   ↓
4. Callback._on_step() is called
   ↓
5. Extract metrics from info
   ↓
6. Log to TensorBoard (real-time graphs)
   ↓
7. Store in episode data (for later plotting)
   ↓
8. If episode ended → _log_episode_metrics()
   ↓
9. Repeat steps 2-8 for next step/episode
   ↓
10. Training ends → Create final plots
```

**That's it!** The callback is your faithful observer, recording everything that happens during training so you can understand, debug, and improve your AI agent's learning process.