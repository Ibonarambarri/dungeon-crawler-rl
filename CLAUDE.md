# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Dungeon Crawler RL Environment** - a custom Gymnasium environment implementing a grid-based dungeon crawler game optimized for tabular RL algorithms. The project is an academic assignment (20% grade weight) for a Reinforcement Learning course.

The environment features:
- 32×32 procedurally generated dungeons with BSP algorithm
- 8×8 logical grid for state encoding (downscaled from visual grid by factor of 4)
- 4 enemies with random movement (not deterministic patrols)
- Combat system with sword/key items, boss fight, and escape objective
- State space: 10,240 theoretical states (64 pos × 4 health × 5 enemy_count × 2 × 2 × 2)
- Three tabular RL algorithm implementations: Q-Learning, SARSA, Expected SARSA

## Core Architecture

### State Space Design (Critical Implementation Detail)

**Visual vs. Logical Grid Split**: The environment uses a dual-grid system to balance visual richness with tractable state spaces:
- **Visual grid**: 32×32 (for pygame rendering)
- **Logical grid**: 8×8 (for RL state encoding)
- **Downscaling**: Agent positions are divided by 4 when encoding states

This design is implemented across two key files:
- `environment/dungeon_env.py` - operates on 32×32 visual grid
- `utils/state_encoder.py` - downscales to 8×8 logical grid via `downscale_factor`

State components encoded as single integer via mixed-radix number system:
- Agent position: 64 positions (8×8)
- Agent health: 4 levels (0-3 HP)
- Enemy count: 5 states (0-4 enemies alive, NOT individual positions)
- Items: has_sword, has_key (binary)
- Boss: boss_alive (binary)

**Note**: We track only the COUNT of enemies alive, not their individual positions. This reduces state space from 694 million to 10,240 states, making tabular RL tractable.

### Environment Architecture (`environment/`)

**`dungeon_env.py`** (700+ lines): Main Gymnasium environment
- Implements full game loop with `reset()`, `step()`, `render()`
- Handles movement, combat, item pickup, enemy AI
- Episode termination: win (boss dead + reach exit), lose (HP=0), timeout (800 steps)
- Reward shaping: 10+ event types with dense intermediate rewards

**`dungeon_generator.py`** (300+ lines): Procedural dungeon generation
- BSP (Binary Space Partitioning) algorithm creates 6 rooms connected by corridors
- Places agent, items (sword/key), enemies (with random starting positions), boss, exit
- Generates unique dungeon layout every episode

**`render_pygame.py`** (500+ lines): PyGame visual renderer
- Camera system centered on agent
- 24×24 pixel cells with health bars, minimap, UI overlay
- Only used when `render_mode='pygame'`

### Agent Implementations (`agents/`)

**Inheritance hierarchy**:
```
BaseTabularAgent (base_agent.py)
├── QLearningAgent (qlearning.py)
├── SARSAAgent (sarsa.py)
└── ExpectedSARSAAgent (expected_sarsa.py)
```

**`base_agent.py`** (200+ lines): Shared functionality
- Sparse Q-table storage using `defaultdict(lambda: np.zeros(n_actions))`
- Epsilon-greedy action selection
- Epsilon decay mechanism
- Save/load via pickle (Q-table + hyperparameters)

**Algorithm-specific files**: Each implements `update()` method with correct TD update rule
- **Q-Learning**: Off-policy, uses `max(Q(s', a))`
- **SARSA**: On-policy, uses `Q(s', a')` where `a'` is actual next action
- **Expected SARSA**: On-policy, uses expected value under epsilon-greedy policy

**Critical SARSA training detail**: SARSA requires selecting the next action BEFORE calling `update()`. See `train.py:205-231` for correct implementation.

### State Encoding (`utils/state_encoder.py`)

**Mixed-radix positional encoding**: Each state component has a "place value" based on cardinality of subsequent components. This ensures bijective (1-to-1) mapping.

Example calculation:
```python
state = (pos_idx * pos_mult +
         health * health_mult +
         enemies_alive * enemies_alive_mult +
         has_sword * sword_mult +
         has_key * key_mult +
         boss_alive * boss_mult)
```

The encoder automatically handles downscaling from 32×32 to 8×8 (divide by 4).

## Common Development Commands

### Training Agents

```bash
# Train Q-Learning agent for 5000 episodes
python train.py --algorithm qlearning --episodes 5000 --run-name qlearning_exp1

# Train SARSA with custom hyperparameters
python train.py --algorithm sarsa --episodes 5000 \
    --alpha 0.2 --gamma 0.99 --epsilon-decay 0.997 \
    --run-name sarsa_custom

# Train Expected SARSA
python train.py --algorithm expected_sarsa --episodes 5000 \
    --run-name expected_sarsa_exp1
```

**Hyperparameter defaults** (recommended for this environment):
- `--alpha 0.1` (learning rate)
- `--gamma 0.95` (discount factor)
- `--epsilon 1.0` (initial exploration)
- `--epsilon-decay 0.995` (reaches ~0.01 by episode 1000)
- `--epsilon-min 0.01` (maintains 1% exploration)

Trained models save to `models/{run_name}/` with checkpoints every 500 episodes.

### Monitoring Training

```bash
# Start TensorBoard to view training metrics
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

**Logged metrics**:
- Episode rewards, success rate, length
- Rolling 100-episode averages
- Q-table statistics (mean/max/min Q-values, num states visited)
- Epsilon decay curve

### Evaluating Agents

```bash
# Evaluate trained agent on 100 episodes
python evaluate.py --model models/qlearning_exp1/final_model.pkl --episodes 100

# Evaluate with text rendering
python evaluate.py --model models/qlearning_exp1/final_model.pkl \
    --episodes 5 --render
```

### Interactive Demo

```bash
# PyGame demo - watch trained agent play with visual rendering
python demo_pygame.py --model models/qlearning_exp1/final_model.pkl --fps 10

# Manual play mode - control the agent yourself
python demo_pygame.py --manual

# Controls: Arrow keys (move), Space (attack), R (reset), M (toggle mode), Q (quit)
```

### Running Tests

```bash
# Test environment functionality
python -m environment.dungeon_env

# Test state encoder
python -m utils.state_encoder

# Test pygame visual rendering
python test_pygame_visual.py
```

## Key Implementation Patterns

### 1. Training Loop Structure (SARSA Special Handling)

SARSA requires selecting the next action during training to use in the update. The training loop handles this:

```python
# For SARSA: select initial action before loop
if algorithm == 'sarsa':
    action = agent.get_action(state, training=True)

while not done:
    # Non-SARSA: select action here
    if algorithm != 'sarsa':
        action = agent.get_action(state, training=True)

    # Take step
    next_state, reward, done = env.step(action)

    # SARSA: select next action for update
    if algorithm == 'sarsa' and not done:
        next_action = agent.get_action(next_state, training=True)
    else:
        next_action = None

    # Update (SARSA uses next_action, others ignore it)
    agent.update(state, action, reward, next_state, next_action)

    # Update for next iteration
    state = next_state
    if algorithm == 'sarsa':
        action = next_action  # Reuse selected action
```

### 2. Sparse Q-Table Storage

Q-tables use `defaultdict(lambda: np.zeros(n_actions))` to only store visited states. This is critical for handling the state space efficiently.

Typical memory usage:
- Theoretical states: 10,240
- Visited states during 5000 episodes: ~3,000-8,000
- Memory: ~1-2 MB per trained agent

### 3. Enemy Movement System

Enemies use random movement (not deterministic patrols). Each step, they randomly choose to move UP/DOWN/LEFT/RIGHT or STAY.

Movement logic from `dungeon_env.py`:
```python
# Random movement: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
move_action = np.random.randint(0, 5)
```

This random movement, combined with only tracking enemy COUNT (not positions), keeps the state space tractable while still providing challenge and variability.

### 4. Reward Shaping Strategy

The environment uses dense reward shaping to guide learning:

| Event | Reward | Purpose |
|-------|--------|---------|
| Valid move | +1.0 | Encourage exploration |
| Step penalty | -0.1 | Encourage efficiency |
| Proximity to objective | ±2.0 per tile | Guide toward current goal |
| Item pickup | +10.0 | Guide to sword/key |
| Kill enemy | +50.0 | Reward combat |
| Kill boss | +100.0 | Major milestone |
| Win (exit) | +500.0 | Ultimate goal |
| Wall hit | -5.0 to -8.0 | Discourage invalid moves |
| Take damage | -10.0 | Avoid enemies |
| Death | -50.0 | Strong penalty |
| **Loop Detection Penalties** | | **Anti-repetition system** |
| Action spam (6+ same) | -15.0 to -20.0 | Prevent single action loops |
| Oscillation (UP-DOWN) | -12.0 | Prevent back-and-forth |
| Position loops (2-3 tiles) | -5.0 to -8.0 | Prevent getting stuck |
| Movement cycles | -10.0 | Prevent circular patterns |

Net reward per valid step: +0.9 (move +1.0 - step penalty -0.1), plus proximity-based shaping.

**Loop Detection System v2 - Multi-Layer Architecture**: The environment uses a sophisticated 5-layer detection system to identify and penalize repetitive behaviors:

**Layer 1: Pattern Repetition Detection**
- Detects repeating action sequences of length 2-5
- Immediate detection after just 2 repetitions (vs 8 steps in v1)
- Penalties: -18 (2-len), -14 (3-len), -10 (4-len), -8 (5-len)

**Layer 2: Action Diversity Analysis**
- Measures variety in last 8 actions
- Only penalizes very low diversity (1-2 unique actions)
- Penalties: -16 (1 action), -10 (2 actions)

**Layer 3: Spatial Loop Detection**
- Tracks position revisitation in last 6 steps
- Only triggers on severe stuck cases (4+ visits to same position)
- Penalty: -10 (very stuck)

**Layer 4: Action Spam Detection**
- Detects same action 6+ consecutive times
- Extra penalty for ATTACK spam
- Penalties: -14 (general), -20 (ATTACK spam)

**Layer 5: Pure Oscillation Detection**
- Perfect alternation between opposite directions (UP-DOWN, LEFT-RIGHT)
- Must be 6+ consecutive perfect pairs
- Penalty: -12

This system achieves **81% test accuracy**, effectively catching obvious loops while allowing legitimate exploration patterns.

## Expected Training Performance

Typical learning progression for 5000 episodes:

**Q-Learning** (off-policy, fastest):
- Episodes 0-500: Random exploration, ~0% success
- Episodes 500-1500: Learns sword value, ~5% success
- Episodes 1500-3000: Strategic item collection, ~20% success
- Episodes 3000-5000: Optimized paths, **40-60% success**

**SARSA** (on-policy, conservative):
- Slower initial learning, more stable final policy
- Better at avoiding risky states
- Expected final success: **30-50%**

**Expected SARSA** (hybrid approach):
- Learning speed similar to Q-Learning
- Lower variance than SARSA
- Expected final success: **35-55%**

## File Organization

```
/
├── environment/          # Gymnasium environment implementation
│   ├── dungeon_env.py         # Main environment (32×32 visual grid)
│   ├── dungeon_generator.py   # BSP procedural generation
│   └── render_pygame.py       # PyGame renderer
├── agents/              # RL algorithm implementations
│   ├── base_agent.py         # Base class with Q-table, epsilon-greedy
│   ├── qlearning.py          # Q-Learning update rule
│   ├── sarsa.py              # SARSA update rule
│   └── expected_sarsa.py     # Expected SARSA update rule
├── utils/               # Utilities
│   └── state_encoder.py      # Dict obs → integer state (8×8 logical)
├── train.py             # Training script with TensorBoard logging
├── evaluate.py          # Evaluation script with statistics
├── demo.py              # Text-based demo with step commentary
├── demo_pygame.py       # Interactive PyGame demo (manual/AI modes)
├── logs/                # TensorBoard logs (created during training)
├── models/              # Saved Q-tables (created during training)
└── requirements.txt     # Dependencies
```

## Important Constraints

1. **Grid size consistency**: If changing `grid_size` in `dungeon_env.py`, also update `state_encoder.py` to match
2. **Enemy count**: `max_enemies` must be consistent between `dungeon_env.py` and `state_encoder.py`
3. **State encoding**: Changes to observation space require updating both encoder multipliers and state space size calculation
4. **SARSA training**: Must follow the special action selection pattern (see training loop above)
5. **Pickle compatibility**: Q-table saving converts `defaultdict` to regular dict; loading recreates `defaultdict`

## Academic Context

This project fulfills **Option B: Environment Design** requirements:
- Custom Gymnasium environment with well-designed state space (10,240 states)
- Tabular RL algorithm implementations (Q-Learning, SARSA, Expected SARSA)
- Comprehensive documentation and analysis
- Complexity justification: tractable yet challenging state space, procedural generation, random enemy movement, multi-objective sequential rewards, strategic planning required

Grade weight: 20% (16% complexity + 4% documentation)
