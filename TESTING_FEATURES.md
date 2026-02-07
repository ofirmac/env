# 🧪 Testing Features Added

## ✨ What's New

I've added comprehensive policy testing to your training script!

## 🎯 Features

### 1. **Automatic Testing After Training** (Default: ON)
- After training completes, automatically tests the best model
- Runs 50 test episodes by default
- Generates detailed statistics and saves to `test_results.json`

### 2. **Standalone Testing Mode**
- Test any saved model without training
- Just provide path to `.zip` model file
- Useful for comparing multiple models

### 3. **Comprehensive Test Metrics**

The testing system provides:

#### 📊 Performance Metrics
- Mean reward ± standard deviation
- Min/max reward range
- Average episode length

#### ⚖️ Balance Metrics
- **Gini coefficient**: Main balance metric (0.0 = perfect, 1.0 = terrible)
- **Phoneme distribution**: How many phonemes each agent spoke
- **Balance metric**: Normalized standard deviation

#### 🎯 Action Analysis
- Visual bar chart of action distribution
- Helps identify if policy is using all available actions
- Shows action counts and percentages

### 4. **Test Results Saved to JSON**
All metrics saved to `test_results.json` for programmatic access

---

## 🚀 Usage Examples

### Production Training (with automatic testing)
```bash
python src/learn/train_env.py --preset production
# Will train for 5000 episodes
# Then automatically test with 50 episodes
# Results in: test_results.json
```

### Custom Test Episodes
```bash
python src/learn/train_env.py --preset production --test-episodes 100
# More test episodes = more reliable statistics
```

### Skip Testing (faster iterations)
```bash
python src/learn/train_env.py --preset dev --no-test
# Good for quick hyperparameter testing
```

### Test Saved Model (no training)
```bash
python src/learn/train_env.py \
    --test-only src/test_result/result/train_production_20260208/best_model/best_model.zip \
    --test-episodes 100
```

### Test Multiple Models
```bash
# Test your best model
python src/learn/train_env.py --test-only path/to/best_model.zip

# Test final model
python src/learn/train_env.py --test-only path/to/final_model.zip

# Test checkpoint
python src/learn/train_env.py --test-only path/to/ppo_guest_100000_steps.zip
```

---

## 📊 Example Output

When testing completes, you'll see:

```
======================================================================
TESTING TRAINED POLICY
======================================================================
Testing model: .../best_model.zip
Test episodes: 50
Deterministic: True

Running test episodes...
  Completed 10/50 episodes...
  Completed 20/50 episodes...
  ...

======================================================================
TEST RESULTS
======================================================================

📊 Performance Metrics:
  Mean reward:     124.56 ± 8.32
  Reward range:    [98.23, 142.87]
  Mean ep length:  500.0 steps

⚖️  Balance Metrics:
  Mean Gini:       0.1234 ± 0.0056
  Gini range:      [0.1102, 0.1387]
  (Lower is better - 0.0 = perfect balance)

🗣️  Phoneme Distribution:
  Agent 0:        812.3 phonemes/episode
  Agent 1:        823.1 phonemes/episode
  Agent 2:        798.6 phonemes/episode

  Balance metric:  0.0345 ± 0.0123
  (Lower is better - 0.0 = perfect balance)

🎯 Action Distribution:
  wait        : ██                                  3.2%
  stare_0     : ████████                           16.5%
  stare_1     : ████████                           15.8%
  stare_2     : ████████                           16.2%
  encourage_0 : ████████                           16.1%
  encourage_1 : ████████                           15.9%
  encourage_2 : ████████                           16.3%

✅ Test results saved to: .../test_results.json
======================================================================
```

---

## 📁 Output Files

### `test_results.json`
Contains all metrics in JSON format:
```json
{
  "num_episodes": 50,
  "mean_reward": 124.56,
  "std_reward": 8.32,
  "mean_gini": 0.1234,
  "mean_phonemes": [812.3, 823.1, 798.6],
  "action_distribution": [0.032, 0.165, 0.158, ...],
  ...
}
```

---

## 🎯 Interpreting Results

### Good Policy Signs ✅
- **Gini < 0.2**: Excellent balance
- **Phoneme distribution similar**: E.g., [800, 820, 790]
- **All actions used**: No action at 0%
- **High mean reward**: Depends on env, but higher is better

### Bad Policy Signs ❌
- **Gini > 0.5**: Poor balance
- **Phoneme distribution skewed**: E.g., [10, 100, 1500]
- **Only 1-2 actions used**: Policy not exploring
- **Low mean reward**: Policy not learning

### Example Interpretation
```
Agent 0: 16.0 phonemes    ← Agent 0 barely speaking!
Agent 1: 146.7 phonemes   ← Agent 1 speaking a bit
Agent 2: 670.0 phonemes   ← Agent 2 dominating!

Mean Gini: 0.5549         ← Poor balance (> 0.5)

Action Distribution:
  stare_2:     51.8%      ← Only focusing on agent 2
  encourage_2: 48.2%      ← Only encouraging agent 2
  others:      0.0%       ← Ignoring other agents!

Diagnosis: Policy learned to only pay attention to agent 2!
```

---

## 🔧 Advanced Usage

### Test with Different Presets
```bash
# Test with production env settings
python src/learn/train_env.py \
    --test-only model.zip \
    --preset production

# Test with dev env settings (shorter episodes)
python src/learn/train_env.py \
    --test-only model.zip \
    --preset dev
```

### Batch Testing
```bash
# Test multiple models
for model in checkpoints/*.zip; do
    echo "Testing $model"
    python src/learn/train_env.py --test-only $model --test-episodes 20
done
```

---

## 💡 Pro Tips

1. **Always test with deterministic=True** (default) for reproducible results
2. **Use 50-100 test episodes** for reliable statistics
3. **Test multiple checkpoints** to see learning progression
4. **Compare test_results.json** files to pick the best model
5. **Low test reward but high training reward?** → Overfitting!
6. **Check action distribution** - if using only 1-2 actions, something's wrong

---

## 🆕 Command Line Arguments Added

```bash
--no-test                  # Skip testing after training
--test-episodes N          # Number of test episodes (default: 50)
--test-only path/to/model  # Test saved model without training
```

---

## ✅ Summary

**Before**: Training finished, but you had no idea how well the policy works
**After**: Comprehensive testing with detailed metrics automatically!

Now every production run gives you:
- ✅ Performance statistics
- ✅ Balance metrics
- ✅ Action analysis
- ✅ JSON results for further analysis
- ✅ Ability to test any saved model anytime

**No more guessing if your policy is good!** 🎉
