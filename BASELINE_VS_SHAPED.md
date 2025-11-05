# Baseline vs Shaped Reward Configuration

## Overview

The project compares two reward configurations for training the DQN agent:

1. **Baseline**: No reward shaping (sparse rewards only)
2. **Shaped**: Reward shaping with progress-based rewards

## Key Difference: Reward Structure

### Baseline Configuration
```python
env_base = CrossyEnv(reward_progress_scale=0.0)
```

**Rewards:**
- Step penalty: **-0.01** per step
- Collision penalty: **-10.0** when hitting a car
- Out-of-bounds penalty: **-50.0** when going outside grid
- Success reward: **+20.0** when reaching finish line
- **No intermediate rewards** for forward progress

**Reward Characteristics:**
- **Sparse rewards**: Only receive significant feedback at terminal states (success/failure)
- Agent must discover the solution through exploration with minimal guidance
- High reward sparsity makes learning slower and more challenging

### Shaped Configuration
```python
env_shaped = CrossyEnv(reward_progress_scale=1.0)  # or 0.5, 1.5, etc.
```

**Rewards:**
- Step penalty: **-0.01** per step
- Collision penalty: **-10.0** when hitting a car
- Out-of-bounds penalty: **-50.0** when going outside grid
- Success reward: **+20.0** when reaching finish line
- **Progress reward**: **+reward_progress_scale × Δz** for each step forward (moving in +z direction)

**Reward Characteristics:**
- **Dense rewards**: Provides immediate positive feedback for forward movement
- Agent receives guidance about which actions lead toward the goal
- Reduces reward sparsity, making learning faster and more efficient

## How Progress Reward Works

```python
# In game_crossy.py step() method:
if self.reward_progress_scale != 0.0 and self._last_player_z is not None:
    delta_z = float(self.player.z - self._last_player_z)
    if delta_z > 0:  # Moving forward (toward finish line)
        reward += self.reward_progress_scale * delta_z
```

**Example:**
- If `reward_progress_scale = 1.0` and agent moves forward by 1 unit:
  - Base step penalty: -0.01
  - Progress reward: +1.0
  - **Net reward: +0.99** (positive feedback for forward movement)

## Performance Comparison

### Training Results (from CHAPTER_7_RESULTS_AND_DISCUSSION.md)

| Metric | Baseline | Reward-Shaped | Improvement |
|--------|----------|---------------|-------------|
| **Success Rate (10k eps)** | 18.5% | 42.3% | **+23.8%** |
| **Mean Reward** | -2.1 | 4.8 | **+6.9** |
| **Collision Rate** | 62% | 48% | **-14%** |
| **Early Success (1-500 eps)** | 0.6-1.4% | 8-15% | **~10× improvement** |
| **Convergence Speed** | Slow | Fast | **2× faster** |

### Learning Characteristics

#### Baseline (No Shaping)
- **Early training (episodes 1-500)**: 0.6-1.4% success rate
- **Converged (episodes 2000-10000)**: 15-25% success rate
- **Mean reward progression**: -12.0 → -1.0 to 2.0
- **Collision rate reduction**: 85% → 50-65%
- **Learning speed**: Slower, requires more exploration

#### Shaped (With Progress Rewards)
- **Early training (episodes 1-500)**: 8-15% success rate (significantly higher)
- **Stable (episodes 2000-10000)**: 35-50% success rate
- **Mean episode reward**: 2.5-8.0 (positive and stable)
- **Convergence**: 2× faster than baseline
- **Learning speed**: Faster, guided exploration

## Why Reward Shaping Works

### 1. **Reduces Reward Sparsity**
- Baseline: Agent only gets meaningful feedback at terminal states
- Shaped: Agent gets continuous feedback for progress

### 2. **Provides Intermediate Guidance**
- Baseline: Agent must discover entire solution path through trial and error
- Shaped: Agent learns that "moving forward is good" early in training

### 3. **Faster Credit Assignment**
- Baseline: Hard to determine which actions led to success (credit assignment problem)
- Shaped: Clear connection between forward movement and positive rewards

### 4. **Better Exploration**
- Baseline: Random exploration may not discover good paths
- Shaped: Progress rewards guide exploration toward promising directions

## Code Implementation

### Experiment Script
The comparison is implemented in `experiment_progress_reward.py`:

```python
def run_experiment(episodes: int = 500, progress_scale: float = 0.5):
    # Baseline: no shaping
    env_base = CrossyEnv(reward_progress_scale=0.0)
    base_stats = train_and_collect(env_base, episodes=episodes, 
                                   save_path="baseline_dqn.pth", label="baseline")

    # Shaped: positive reward for forward progress
    env_shaped = CrossyEnv(reward_progress_scale=progress_scale)
    shaped_stats = train_and_collect(env_shaped, episodes=episodes,
                                     save_path="shaped_dqn.pth", label="shaped")
```

### Running the Experiment
```bash
python experiment_progress_reward.py
```

This will:
1. Train baseline model (no reward shaping)
2. Train shaped model (with progress rewards)
3. Generate comparison plots (`baseline_metrics.png` vs `shaped_metrics.png`)
4. Save statistics to CSV files (`baseline_stats.csv` vs `shaped_stats.csv`)

## Output Files

After running the experiment, you'll find:

```
models_compressed/
├── baseline_dqn.pth          # Trained baseline model
├── shaped_dqn.pth            # Trained shaped model
├── baseline_metrics.png      # Baseline training metrics plot
├── shaped_metrics.png        # Shaped training metrics plot
├── baseline_stats.csv        # Baseline episode statistics
└── shaped_stats.csv          # Shaped episode statistics
```

## Summary

| Aspect | Baseline | Shaped |
|--------|----------|--------|
| **Reward Density** | Sparse (only terminal states) | Dense (progress rewards) |
| **Learning Speed** | Slow | Fast (2× faster) |
| **Early Performance** | 0.6-1.4% success | 8-15% success |
| **Final Performance** | 15-25% success | 35-50% success |
| **Mean Reward** | Negative to slightly positive | Positive and stable |
| **Use Case** | Baseline comparison, research | Production training |

**Recommendation**: Use **shaped rewards** for training, as they provide significantly better learning efficiency and final performance. Use baseline for comparison and understanding the impact of reward shaping.

