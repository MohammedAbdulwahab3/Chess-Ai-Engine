"""
Replay Buffer for RL Training

Stores game experiences for training.
Ensures correct tensor shapes and types.
"""

import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action_idx, value_target):
        """
        Save a transition
        state: board tensor (12, 8, 8)
        action_idx: int
        value_target: float
        """
        self.buffer.append((state, action_idx, value_target))
        
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        
        states, actions, values = zip(*batch)
        
        # Convert to tensors
        # states: List of tensors -> (B, 12, 8, 8)
        states_tensor = torch.stack(states)
        
        # actions: List of ints -> (B,)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        
        # values: List of floats -> (B, 1)
        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        
        return states_tensor, actions_tensor, values_tensor
        
    def __len__(self):
        return len(self.buffer)
