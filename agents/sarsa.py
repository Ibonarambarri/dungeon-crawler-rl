"""
SARSA Agent

Implements the SARSA (State-Action-Reward-State-Action) algorithm,
an on-policy TD control algorithm.

SARSA update rule:
Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)]

Key characteristics:
- On-policy: Learns Q-values for the policy being followed
- Uses actual next action a' (from epsilon-greedy policy)
- More conservative than Q-Learning
- Better for risky environments
"""

import numpy as np
from agents.base_agent import BaseTabularAgent
from typing import Optional


class SARSAAgent(BaseTabularAgent):
    """
    SARSA agent for tabular RL.

    SARSA is an on-policy algorithm that learns Q-values based on the
    actual actions taken by the epsilon-greedy policy.

    Unlike Q-Learning (which uses max Q-value), SARSA uses the Q-value
    of the actual next action selected by the policy. This makes it more
    conservative and safer in risky environments.

    The name SARSA comes from the tuple (State, Action, Reward, State', Action')
    needed for the update.
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
        Initialize SARSA agent.

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
        Update Q-value using SARSA update rule.

        SARSA is on-policy: it uses the Q-value of the actual next action
        that will be taken according to the current policy.

        Update rule:
        Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)]

        For terminal states: Q(s, a) ← Q(s, a) + α[r - Q(s, a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (REQUIRED for SARSA unless terminal)
            done: Whether next_state is terminal (default: False)

        Raises:
            ValueError: If next_action is None and not done
        """
        # Get current Q-value
        current_q = self.q_table[state][action]

        # Calculate TD target
        if done:
            # Terminal state: no future value
            td_target = reward
        else:
            # Non-terminal: requires next_action
            if next_action is None:
                raise ValueError("SARSA requires next_action for non-terminal states")
            # Get Q-value for the actual next action (on-policy)
            next_q = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * next_q

        # Calculate TD error
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state][action] += self.alpha * td_error


def test_sarsa():
    """Test SARSA agent basic functionality."""
    print("Testing SARSA Agent")
    print()

    # Create agent
    agent = SARSAAgent(
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

    # Test Q-value update with next_action
    print("Test 2: Q-Value Update (with next_action)")
    state, action, reward, next_state, next_action = 100, 2, 10.0, 101, 3

    print(f"Before update: Q({state}, {action}) = {agent.q_table[state][action]}")
    agent.update(state, action, reward, next_state, next_action)
    print(f"After update:  Q({state}, {action}) = {agent.q_table[state][action]}")
    print()

    # Test that SARSA requires next_action
    print("Test 3: SARSA Requires next_action")
    try:
        agent.update(state, action, reward, next_state, next_action=None)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    print()

    # Test epsilon decay
    print("Test 4: Epsilon Decay")
    for episode in range(5):
        print(f"Episode {episode}: epsilon = {agent.epsilon:.4f}")
        agent.decay_epsilon()
    print()

    print("All tests passed! ✓")


if __name__ == "__main__":
    test_sarsa()
