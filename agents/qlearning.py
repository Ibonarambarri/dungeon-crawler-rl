"""
Q-Learning Agent

Implements the Q-Learning algorithm, an off-policy TD control algorithm.

Q-Learning update rule:
Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]

Key characteristics:
- Off-policy: Learns optimal policy while following epsilon-greedy policy
- Uses max Q-value of next state (optimistic)
- Tends to learn faster but may be more aggressive
"""

import numpy as np
from agents.base_agent import BaseTabularAgent
from typing import Optional


class QLearningAgent(BaseTabularAgent):
    """
    Q-Learning agent for tabular RL.

    Q-Learning is an off-policy algorithm that learns the optimal Q-function
    by always using the maximum Q-value of the next state, regardless of the
    action actually taken.

    This makes it more optimistic and aggressive compared to SARSA, which can
    lead to faster learning but potentially riskier behavior during training.
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998,  # Changed from 0.995 for slower decay
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning agent.

        Args:
            n_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        super().__init__(
            n_actions=n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int] = None,
        done: bool = False
    ) -> None:
        """
        Update Q-value using Q-Learning update rule.

        Q-Learning is off-policy: it always uses the maximum Q-value of the
        next state, regardless of which action will actually be taken.

        Update rule:
        Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]

        For terminal states: Q(s, a) ← Q(s, a) + α[r - Q(s, a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Not used (Q-Learning doesn't need it)
            done: Whether next_state is terminal (default: False)
        """
        # Get current Q-value
        current_q = self.q_table[state][action]

        # Calculate TD target
        if done:
            # Terminal state: no future value
            td_target = reward
        else:
            # Non-terminal: use max Q-value for next state (off-policy)
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q

        # Calculate TD error
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state][action] += self.alpha * td_error


def test_qlearning():
    """Test Q-Learning agent basic functionality."""
    print("Testing Q-Learning Agent")
    print()

    # Create agent
    agent = QLearningAgent(
        n_actions=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0
    )

    print(f"Agent: {agent}")
    print()

    # Test action selection
    print("Test 1: Action Selection")
    state = 100
    actions = [agent.get_action(state, training=True) for _ in range(10)]
    print(f"10 actions from state {state}: {actions}")
    print(f"(Should be random with epsilon=1.0)")
    print()

    # Test Q-value update
    print("Test 2: Q-Value Update")
    state, action, reward, next_state = 100, 2, 10.0, 101

    print(f"Before update: Q({state}, {action}) = {agent.q_table[state][action]}")
    agent.update(state, action, reward, next_state)
    print(f"After update:  Q({state}, {action}) = {agent.q_table[state][action]}")
    print()

    # Test epsilon decay
    print("Test 3: Epsilon Decay")
    for episode in range(5):
        print(f"Episode {episode}: epsilon = {agent.epsilon:.4f}")
        agent.decay_epsilon()
    print()

    # Test Q-table stats
    print("Test 4: Q-Table Statistics")
    # Add some more Q-values
    for s in range(100, 105):
        for a in range(5):
            agent.q_table[s][a] = np.random.randn()

    stats = agent.get_q_table_stats()
    print(f"Q-table stats: {stats}")
    print()

    print("All tests passed! ✓")


if __name__ == "__main__":
    test_qlearning()
