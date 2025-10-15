"""
Base Tabular RL Agent

Provides base functionality for tabular RL algorithms including:
- Q-table management (using sparse defaultdict)
- Epsilon-greedy action selection
- Epsilon decay
- Model saving/loading

All specific algorithms (Q-Learning, SARSA, Expected SARSA) inherit from this class
and implement their own update() method.
"""

import numpy as np
import pickle
from collections import defaultdict
from typing import Optional, Dict, Any
import os


class BaseTabularAgent:
    """
    Base class for tabular RL agents.

    This class provides common functionality for all tabular RL algorithms:
    - Q-table as sparse dictionary (defaultdict)
    - Epsilon-greedy policy
    - Epsilon decay mechanism
    - Save/load functionality

    Attributes:
        n_actions (int): Number of possible actions
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Current exploration rate
        epsilon_decay (float): Epsilon decay multiplier per episode
        epsilon_min (float): Minimum epsilon value
        q_table (defaultdict): Sparse Q-table mapping states to action values
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize the base tabular agent.

        Args:
            n_actions: Number of possible actions
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate (1.0 = full exploration)
            epsilon_decay: Multiplicative decay for epsilon per episode
            epsilon_min: Minimum epsilon value (maintains exploration)
        """
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> array of Q-values for each action
        # Using defaultdict for sparse representation (only stores visited states)
        self.q_table: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )

        # Statistics
        self.states_visited = set()

    def get_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        During training, explores with probability epsilon.
        During evaluation, always exploits (greedy).

        Args:
            state: Current state (integer index)
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: Selected action (0 to n_actions-1)
        """
        self.states_visited.add(state)

        # Epsilon-greedy during training, greedy during evaluation
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to Q-table
            q_values = self.q_table[state]
            # Break ties randomly
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int] = None
    ) -> None:
        """
        Update Q-value based on experience.

        This is an abstract method that must be implemented by subclasses.
        Different algorithms (Q-Learning, SARSA, Expected SARSA) will
        implement this differently.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (only needed for SARSA)
        """
        raise NotImplementedError("Subclasses must implement update()")

    def decay_epsilon(self) -> None:
        """
        Decay epsilon after each episode.

        Reduces exploration over time as agent learns.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """
        Save agent's Q-table and parameters to file.

        Args:
            filepath: Path to save file (will create directory if needed)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert defaultdict to regular dict for pickling
        save_data = {
            'q_table': dict(self.q_table),
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'states_visited': self.states_visited
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Agent saved to {filepath}")
        print(f"  States visited: {len(self.states_visited):,}")
        print(f"  Q-table size: {len(self.q_table):,}")

    def load(self, filepath: str) -> None:
        """
        Load agent's Q-table and parameters from file.

        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Restore Q-table as defaultdict
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            save_data['q_table']
        )

        # Restore parameters
        self.n_actions = save_data['n_actions']
        self.alpha = save_data['alpha']
        self.gamma = save_data['gamma']
        self.epsilon = save_data['epsilon']
        self.epsilon_decay = save_data['epsilon_decay']
        self.epsilon_min = save_data['epsilon_min']
        self.states_visited = save_data.get('states_visited', set())

        print(f"Agent loaded from {filepath}")
        print(f"  States visited: {len(self.states_visited):,}")
        print(f"  Q-table size: {len(self.q_table):,}")
        print(f"  Current epsilon: {self.epsilon:.4f}")

    def get_q_table_stats(self) -> Dict[str, float]:
        """
        Get statistics about the Q-table.

        Returns:
            dict: Statistics including mean, max, min Q-values
        """
        if len(self.q_table) == 0:
            return {
                'mean_q': 0.0,
                'max_q': 0.0,
                'min_q': 0.0,
                'num_states': 0
            }

        all_q_values = []
        for state_q_values in self.q_table.values():
            all_q_values.extend(state_q_values)

        all_q_values = np.array(all_q_values)

        return {
            'mean_q': float(np.mean(all_q_values)),
            'max_q': float(np.max(all_q_values)),
            'min_q': float(np.min(all_q_values)),
            'num_states': len(self.q_table)
        }

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"{self.__class__.__name__}("
                f"alpha={self.alpha}, "
                f"gamma={self.gamma}, "
                f"epsilon={self.epsilon:.4f}, "
                f"states_visited={len(self.states_visited)})")
