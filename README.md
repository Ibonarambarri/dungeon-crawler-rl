# Dungeon Crawler RL Environment

A minimal custom Gymnasium environment for tabular reinforcement learning algorithms, featuring an 8×8 grid-based dungeon navigation task with sparse rewards.

## 🎯 Project Overview

This project implements a simplified dungeon crawler environment optimized for tabular RL methods (Q-Learning, SARSA, Expected SARSA). The agent must learn to collect a key and reach an exit door through pure exploration with sparse reward signals.

### Key Features

- **Ultra-simplified 8×8 grid** with global vision
- **128 state space** (64 positions × 2 key states)
- **Sparse rewards only** - no reward shaping
- **Three tabular RL algorithms**: Q-Learning, SARSA, Expected SARSA
- **PyGame visualization** for interactive demos
- **TensorBoard logging** for training analysis

## 🏗️ Architecture

### Environment (`environment/`)

- **dungeon_env.py**: Main Gymnasium environment (8×8 grid, 100 max steps)
- **dungeon_generator.py**: Procedural dungeon generation with border walls
- **render_pygame.py**: PyGame renderer with 48×48 pixel cells

### Agents (`agents/`)

- **base_agent.py**: Base class with sparse Q-table, epsilon-greedy policy
- **qlearning.py**: Q-Learning (off-policy)
- **sarsa.py**: SARSA (on-policy)
- **expected_sarsa.py**: Expected SARSA (hybrid)

### State Encoding (`utils/`)

- **state_encoder.py**: Converts 8×8 observations to integer states (0-127)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dungeon-crawler-rl.git
cd dungeon-crawler-rl

# Install dependencies
pip install -r requirements.txt
```

### Training an Agent

```bash
# Train Q-Learning for 2000 episodes
python train.py --algorithm qlearning --episodes 2000 --run-name ql_8x8

# Train SARSA with custom hyperparameters
python train.py --algorithm sarsa --episodes 2000 --alpha 0.2 --gamma 0.99

# Train Expected SARSA
python train.py --algorithm expected_sarsa --episodes 2000
```

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# Open http://localhost:6006 in your browser
```

### Evaluating a Trained Agent

```bash
# Evaluate on 100 episodes
python evaluate.py --model models/ql_8x8/final_model.pkl --episodes 100
```

### Interactive Demo

```bash
# Manual play mode
python demo_pygame.py --manual

# Watch trained agent play
python demo_pygame.py --model models/ql_8x8/final_model.pkl
```

## 🎮 Environment Details

### State Space

- **Total states**: 128
- **Agent position**: 64 possibilities (8×8 grid)
- **Key possession**: 2 states (has_key: 0 or 1)
- **Encoding**: `state = pos_idx * 2 + has_key`

### Action Space

4 discrete movement actions:
- 0: Move UP
- 1: Move DOWN  
- 2: Move LEFT
- 3: Move RIGHT

### Reward Structure (Sparse Only)

- **+10.0** - Collect key
- **+100.0** - Reach door with key (victory)
- **0.0** - All other transitions

## 📊 Expected Performance

Typical learning curves for 2000 episodes:

- **Q-Learning**: 30-50% success rate
- **SARSA**: 25-45% success rate
- **Expected SARSA**: 30-50% success rate

## 🔧 Hyperparameters

Default values:

```python
alpha = 0.1              # Learning rate
gamma = 0.95             # Discount factor
epsilon = 1.0            # Initial exploration
epsilon_decay = 0.995    # Decay per episode
epsilon_min = 0.01       # Minimum exploration
```

## 📝 Academic Context

Developed as part of a Reinforcement Learning course assignment, implementing custom Gymnasium environment with tabular RL algorithms.

## 📄 License

Available for educational purposes.
