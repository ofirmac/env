# Professional RL Training Guide

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default settings
python src/learn/train_env.py

# Quick development test (100 episodes, faster)
python src/learn/train_env.py --preset dev

# Fast training (500 episodes, 4 parallel envs)
python src/learn/train_env.py --preset fast

# Production training (5000 episodes, best quality)
python src/learn/train_env.py --preset production
```

## 📋 Command Line Options

### Presets
```bash
--preset dev          # Fast iterations for development (100 episodes)
--preset fast         # Quick training (500 episodes, 4 parallel envs)
--preset production   # Thorough training (5000 episodes, best results)
--preset default      # Standard settings (1000 episodes)
```

### Custom Training
```bash
--episodes 2000                    # Set total episodes
--parallel-envs 8                  # Use 8 parallel environments
--learning-rate 0.0001             # Set learning rate
--batch-size 512                   # Set batch size
--seed 123                         # Set random seed
```

### Paths & Resume
```bash
--results-dir my_experiment        # Custom results directory
--resume-from path/to/model.zip    # Resume training from saved model
```

### Flags
```bash
--no-tensorboard                   # Disable TensorBoard logging
--reward-shaping                   # Enable reward shaping
--no-test                          # Skip testing after training
--test-episodes 100                # Number of test episodes (default: 50)
```

### Testing
```bash
# Test a trained model (without training)
--test-only path/to/model.zip      # Test saved model with 50 episodes
--test-only path/to/model.zip --test-episodes 100  # Test with custom episodes
```

## 📊 Examples

### Example 1: Quick Development Test
```bash
python src/learn/train_env.py \
    --preset dev \
    --episodes 50
```

### Example 2: Production Training with Custom Settings
```bash
python src/learn/train_env.py \
    --preset production \
    --parallel-envs 8 \
    --learning-rate 0.0001 \
    --results-dir my_best_model
```

### Example 3: Resume Training
```bash
python src/learn/train_env.py \
    --resume-from src/test_result/train_default_20260208/best_model/best_model.zip \
    --episodes 1000
```

### Example 4: Fast Training Without TensorBoard
```bash
python src/learn/train_env.py \
    --preset fast \
    --no-tensorboard \
    --parallel-envs 8
```

### Example 5: Production Training with Custom Test Episodes
```bash
python src/learn/train_env.py \
    --preset production \
    --test-episodes 100
# Will train AND test with 100 episodes after training
```

### Example 6: Test Only (No Training)
```bash
# Test your best model
python src/learn/train_env.py \
    --test-only src/test_result/result/train_production_20260208/best_model/best_model.zip \
    --test-episodes 100

# Quick test with 10 episodes
python src/learn/train_env.py \
    --test-only path/to/model.zip \
    --test-episodes 10
```

### Example 7: Training Without Post-Training Test
```bash
python src/learn/train_env.py \
    --preset production \
    --no-test
# Skips automatic testing after training
```

## 📁 Output Structure

```
src/test_result/result/train_default_20260208_143022/
├── config.json                 # All configuration saved here
├── test_results.json           # Post-training test results ✨
├── best_model/                 # Best model (by evaluation reward)
│   └── best_model.zip
├── final_model_*.zip           # Final trained model
├── checkpoints/                # Periodic checkpoints
│   ├── ppo_guest_50000_steps.zip
│   ├── ppo_guest_100000_steps.zip
│   └── ...
├── logs/                       # Detailed episode logs
│   ├── episode_data.pkl
│   └── eval/
└── tensorboard/                # TensorBoard logs
    └── PPO_GuestEnv_1/
```

## 📈 Monitor Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir src/test_result/train_default_20260208_143022/tensorboard

# Open browser to http://localhost:6006
```

### View Logs
```bash
# Check episode logs
cat src/test_result/result/train_default_20260208_143022/logs/episode_data.pkl

# Check configuration
cat src/test_result/result/train_default_20260208_143022/config.json

# View test results (JSON format)
cat src/test_result/result/train_default_20260208_143022/test_results.json
```

## 🧪 Understanding Test Results

After training (or using `--test-only`), you'll get comprehensive test statistics:

### Performance Metrics
- **Mean reward**: Average reward across test episodes
- **Reward range**: Best and worst episode rewards
- **Mean episode length**: Average steps per episode

### Balance Metrics
- **Mean Gini coefficient**: 0.0 = perfect balance, 1.0 = complete imbalance
- **Phoneme distribution**: How many phonemes each agent spoke
- **Balance metric**: Normalized standard deviation (lower = better balance)

### Action Distribution
- **Visual bar chart**: Shows which actions the policy prefers
- Helps understand if policy is using all actions or stuck on a few

### Example Test Output
```
📊 Performance Metrics:
  Mean reward:     89.01 ± 1.73
  Reward range:    [87.38, 91.40]
  Mean ep length:  200.0 steps

⚖️  Balance Metrics:
  Mean Gini:       0.5549 ± 0.0086
  (Lower is better - 0.0 = perfect balance)

🗣️  Phoneme Distribution:
  Agent 0:        16.0 phonemes/episode
  Agent 1:        146.7 phonemes/episode
  Agent 2:        670.0 phonemes/episode

🎯 Action Distribution:
  wait        :                                  0.0%
  stare_2     : █████████████████████████       51.8%
  encourage_2 : ████████████████████████        48.2%
```

## 🎯 Configuration Presets Comparison

| Preset       | Episodes | Steps | Batch | Envs | Speed    | Quality  |
|--------------|----------|-------|-------|------|----------|----------|
| **dev**      | 100      | 200   | 128   | 1    | ⚡⚡⚡    | ⭐       |
| **fast**     | 500      | 500   | 128   | 4    | ⚡⚡      | ⭐⭐     |
| **default**  | 1000     | 500   | 256   | 1    | ⚡       | ⭐⭐⭐   |
| **production** | 5000   | 600   | 256   | 4    | 🐌       | ⭐⭐⭐⭐ |

## 💡 Tips

### Training Tips
1. **Start with `--preset dev`** for quick testing (takes 5 minutes)
2. **Use `--preset fast`** for iterating on hyperparameters
3. **Use `--preset production`** for final training runs (2-4 hours)
4. **Use `--parallel-envs 4-8`** for faster training (if you have CPU cores)
5. **Monitor with TensorBoard** to track progress in real-time
6. **Save important runs** - copy results directory to safe location

### Testing Tips
7. **Test after training** is enabled by default (50 episodes)
8. **Use `--test-episodes 100`** for more reliable statistics
9. **Use `--test-only`** to quickly test saved models
10. **Check `test_results.json`** for detailed metrics in JSON format
11. **Good Gini < 0.2** means excellent balance, **Gini > 0.5** means poor balance
12. **Use `--no-test`** to skip testing if you're just iterating quickly

## 🔧 Advanced: Edit Configuration in Code

For fine-tuned control, edit the dataclasses in `train_env.py`:

```python
@dataclass
class EnvConfig:
    max_steps: int = 500
    reward_shaping: bool = False
    # ... modify these

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    # ... modify these
```

## 🆘 Troubleshooting

### Out of Memory
- Reduce `--parallel-envs`
- Reduce `--batch-size`
- Use `--no-tensorboard`

### Training Too Slow
- Increase `--parallel-envs`
- Use `--preset fast`
- Use `--no-tensorboard`

### Not Learning Well
- Increase `--episodes`
- Try `--learning-rate 0.0001`
- Use `--preset production`
- Enable `--reward-shaping`
