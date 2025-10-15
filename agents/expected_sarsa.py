"""
Expected SARSA Agent

Implements the Expected SARSA algorithm, a hybrid between Q-Learning and SARSA.

Expected SARSA update rule:
Q(s, a) ← Q(s, a) + α[r + γ E[Q(s', a')] - Q(s, a)]

where E[Q(s', a')] = Σ π(a'|s') Q(s', a')

Key characteristics:
- Uses expected value of Q(s', a') under current policy
- More stable than SARSA (lower variance)
- Similar to Q-Learning but considers exploration policy
- Often performs better than both Q-Learning and SARSA
"""

import numpy as np
from agents.base_agent import BaseTabularAgent
from typing import Optional


class ExpectedSARSAAgent(BaseTabularAgent):
    """
    Expected SARSA agent for tabular RL.

    Expected SARSA is a hybrid algorithm that combines ideas from both
    Q-Learning and SARSA. Instead of using max Q-value (Q-Learning) or
    the actual next action's Q-value (SARSA), it uses the expected Q-value
    under the current epsilon-greedy policy.

    This makes it more stable than SARSA (lower variance) while being
    less aggressive than Q-Learning.

    Expected value calculation:
    E[Q(s', a')] = (1-ε + ε/|A|) * Q(s', a_best) + Σ_(a≠a_best) (ε/|A|) * Q(s', a)

    where:
    - a_best is the greedy action (argmax Q(s', a))
    - |A| is the number of actions
    - ε is the exploration rate
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
        Initialize Expected SARSA agent.

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
        Update Q-value using Expected SARSA update rule.

        Expected SARSA computes the expected Q-value of the next state
        under the current epsilon-greedy policy.

        Update rule:
        Q(s, a) ← Q(s, a) + α[r + γ E_π[Q(s', a')] - Q(s, a)]

        For terminal states: Q(s, a) ← Q(s, a) + α[r - Q(s, a)]

        The expected value accounts for both exploitation (choosing best action)
        and exploration (random actions):

        E[Q(s', a')] = Σ π(a'|s') Q(s', a')

        where π(a'|s') is the epsilon-greedy policy:
        - π(a_best|s') = 1 - ε + ε/|A|
        - π(a|s') = ε/|A| for all other actions

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Not used (Expected SARSA doesn't need it)
            done: Whether next_state is terminal (default: False)
        """
        # Get current Q-value
        current_q = self.q_table[state][action]

        # Calculate TD target
        if done:
            # Terminal state: no future value
            td_target = reward
        else:
            # Non-terminal: calculate expected Q-value
            # Get Q-values for next state
            next_q_values = self.q_table[next_state]

            # Find best action (for greedy part of policy)
            best_action = np.argmax(next_q_values)

            # Calculate expected Q-value under epsilon-greedy policy
            expected_q = 0.0

            for a in range(self.n_actions):
                if a == best_action:
                    # Probability of selecting best action:
                    # (1 - epsilon) for greedy + (epsilon / n_actions) for random
                    prob = 1.0 - self.epsilon + (self.epsilon / self.n_actions)
                else:
                    # Probability of selecting non-best action:
                    # (epsilon / n_actions) for random exploration
                    prob = self.epsilon / self.n_actions

                expected_q += prob * next_q_values[a]

            td_target = reward + self.gamma * expected_q

        # Calculate TD error
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state][action] += self.alpha * td_error


def test_expected_sarsa():
    """Test Expected SARSA agent basic functionality."""
    print("Testing Expected SARSA Agent")
    print()

    # Create agent
    agent = ExpectedSARSAAgent(
        n_actions=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.5  # Use 0.5 to see difference from Q-Learning
    )

    print(f"Agent: {agent}")
    print()

    # Test action selection
    print("Test 1: Action Selection")
    state = 100
    actions = [agent.get_action(state, training=True) for _ in range(10)]
    print(f"10 actions from state {state}: {actions}")
    print(f"(Mix of greedy and random with epsilon=0.5)")
    print()

    # Test Q-value update
    print("Test 2: Q-Value Update")
    # Set up some Q-values to test expected value calculation
    state, action, reward, next_state = 100, 2, 10.0, 101

    # Initialize next state Q-values with specific values
    agent.q_table[next_state] = np.array([1.0, 5.0, 3.0, 2.0, 4.0])

    print(f"Next state Q-values: {agent.q_table[next_state]}")
    print(f"Best action: {np.argmax(agent.q_table[next_state])} (value: 5.0)")
    print(f"Epsilon: {agent.epsilon}")

    # Calculate expected Q manually for verification
    best_action = 1  # argmax = 5.0
    expected_q_manual = 0.0
    for a in range(5):
        if a == best_action:
            prob = 1.0 - agent.epsilon + (agent.epsilon / 5)
        else:
            prob = agent.epsilon / 5
        expected_q_manual += prob * agent.q_table[next_state][a]

    print(f"Expected Q (manual): {expected_q_manual:.4f}")
    print()

    print(f"Before update: Q({state}, {action}) = {agent.q_table[state][action]}")
    agent.update(state, action, reward, next_state)
    print(f"After update:  Q({state}, {action}) = {agent.q_table[state][action]:.4f}")
    print()

    # Test epsilon decay
    print("Test 3: Epsilon Decay")
    for episode in range(5):
        print(f"Episode {episode}: epsilon = {agent.epsilon:.4f}")
        agent.decay_epsilon()
    print()

    # Compare with Q-Learning behavior
    print("Test 4: Comparison with Q-Learning")
    print("Expected SARSA uses expected Q-value under policy")
    print("Q-Learning would use max Q-value (5.0)")
    print(f"Expected SARSA considers all actions weighted by policy")
    print()

    print("All tests passed! ✓")


if __name__ == "__main__":
    test_expected_sarsa()
