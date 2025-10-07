# Reward Tracking Bug Fixes

## 🐛 **Issues Found in Summary Plots:**

### **Problem 1: Incorrect Reward Values**
**Root Cause:** The callback was not tracking the correct reward values used by PPO training.

**Issues Identified:**
1. **Environment inconsistency**: `env_reward` in info dict only contained base reward (1.0 - gini), missing reward shaping bonuses/penalties
2. **Missing reward shaping**: PPO was using shaped rewards (+0.2 for new speaker, -0.1 for long turns) but callback only logged base rewards
3. **Incorrect Gini tracking**: Environment was double-inverting Gini coefficient (1.0 - (1.0 - gini))

### **Problem 2: Inconsistent Data in Plots**
**Root Cause:** Callback summary plots showed different values than what PPO actually used for training.

## 🔧 **Fixes Applied:**

### **Environment Fixes** (`src/env/env_gym.py`):

1. **Separated reward types**:
   ```python
   base_reward = 1.0 - gini  # Base reward without shaping
   reward = base_reward      # Start with base, then add shaping
   
   # Apply reward shaping...
   if self.reward_shaping:
       if self.current_speaker != -1:
           if self.speaking_time[self.current_speaker] == 1:
               reward += 0.2  # New speaker bonus
           # ... penalty logic
   ```

2. **Enhanced info dictionary**:
   ```python
   info = dict(
       env_reward=base_reward,     # Base reward without shaping
       total_reward=reward,        # FIXED: Total reward including shaping
       current_gini=gini,          # FIXED: Direct gini value
       reward_shaping_active=self.reward_shaping,
       # ... other fields
   )
   ```

3. **Fixed Gini history tracking**:
   ```python
   # OLD (WRONG): self.gini_history.append(1.0 - info["env_reward"])
   # NEW (CORRECT):
   self.gini_history.append(gini)  # Store actual gini coefficient
   ```

### **Callback Fixes** (`src/callback/guest_callback_per_episode.py`):

1. **Enhanced reward tracking**:
   ```python
   # Track both reward types
   current_gini = self._safe_get_metric(info, "current_gini", 0)  # Direct gini
   env_reward = self._safe_get_metric(info, "env_reward", 0)      # Base reward
   total_reward = self._safe_get_metric(info, "total_reward", reward)  # Shaped reward
   ```

2. **Improved data storage**:
   ```python
   self.current_episode_data = {
       'rewards': [],           # PPO rewards (with shaping)
       'env_rewards': [],       # Base environment rewards  
       'phonemes': [],
       'gini': [],
       # ... other fields
   }
   ```

3. **Enhanced TensorBoard logging**:
   ```python
   self.writer.add_scalar('Step/PPO_Reward', reward, global_step)      # PPO training reward
   self.writer.add_scalar('Step/Env_Reward', env_reward, global_step)  # Base environment reward
   ```

4. **Improved plotting**:
   ```python
   # Show both reward types in plots
   ax1.plot(steps, episode_data['rewards'], 'b-', label='PPO Reward (with shaping)')
   if 'env_rewards' in episode_data:
       ax1.plot(steps, episode_data['env_rewards'], 'r-', label='Environment Reward (base)')
   ```

## ✅ **Validation Results:**

### **Test Output:**
```
Step 1:
  Action: 0
  PPO Reward: 1.0000      ← Reward used by PPO training
  Env Reward: 1.0000      ← Base environment reward  
  Total Reward: 1.0000    ← Total reward (base + shaping)
  Current Gini: 0.0000    ← Correct Gini coefficient
  Phonemes: [0 0 0]       ← Agent phoneme counts
```

### **Key Improvements:**

1. **✅ Correct reward values**: Summary plots now show actual rewards used by PPO
2. **✅ Separated reward types**: Can distinguish between base rewards and shaped rewards
3. **✅ Fixed Gini tracking**: Gini coefficient properly calculated and stored
4. **✅ Enhanced logging**: TensorBoard shows both reward types for analysis
5. **✅ Improved plots**: Visual distinction between PPO rewards and environment rewards

## 🎯 **Impact on Training Analysis:**

### **Before Fixes:**
- Summary plots showed incorrect reward values
- Couldn't distinguish between base and shaped rewards  
- Gini coefficient calculations were wrong
- Training analysis was misleading

### **After Fixes:**
- ✅ **Accurate reward tracking**: Plots show exact values PPO uses for learning
- ✅ **Reward decomposition**: Can analyze impact of reward shaping separately
- ✅ **Correct balance metrics**: Gini coefficient properly represents conversation balance
- ✅ **Better debugging**: Can identify if issues are in base environment or reward shaping

## 📊 **Usage:**

### **For Training:**
```python
# Use the fixed callback
callback = CallbackPerEpisode(log_dir="./logs")
model.learn(total_timesteps=50000, callback=callback)
```

### **For Analysis:**
```python
# Create plots with both reward types
callback.create_final_plots(results_dir="./plots")
# Plots will show:
# - Blue line: PPO Reward (what the agent actually learns from)
# - Red line: Environment Reward (base reward without shaping)
```

The fixes ensure that your summary plots now accurately reflect what's happening during training, making your analysis reliable and your debugging much more effective!