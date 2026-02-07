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

## 📁 Output Structure

```
src/test_result/train_default_20260208_143022/
├── config.json                 # All configuration saved here
├── best_model/                 # Best model (by evaluation reward)
│   └── best_model.zip
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
cat src/test_result/train_default_20260208_143022/logs/episode_data.pkl

# Check configuration
cat src/test_result/train_default_20260208_143022/config.json
```

## 🎯 Configuration Presets Comparison

| Preset       | Episodes | Steps | Batch | Envs | Speed    | Quality  |
|--------------|----------|-------|-------|------|----------|----------|
| **dev**      | 100      | 200   | 128   | 1    | ⚡⚡⚡    | ⭐       |
| **fast**     | 500      | 500   | 128   | 4    | ⚡⚡      | ⭐⭐     |
| **default**  | 1000     | 500   | 256   | 1    | ⚡       | ⭐⭐⭐   |
| **production** | 5000   | 600   | 256   | 4    | 🐌       | ⭐⭐⭐⭐ |

## 💡 Tips

1. **Start with `--preset dev`** for quick testing
2. **Use `--preset fast`** for iterating on hyperparameters
3. **Use `--preset production`** for final training runs
4. **Use `--parallel-envs 4-8`** for faster training (if you have CPU cores)
5. **Monitor with TensorBoard** to track progress
6. **Save important runs** - copy results directory to safe location

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
