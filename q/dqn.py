import numpy as np
import torch
from collections import deque
from q.model import CustomModel
from torch.utils.data import Dataset
from torch import Tensor
import torch.nn as nn

class Dqn:
    def __init__(self, state_dim: int = 8, action_dim: int = 5, replay_buffer_size: int = 1000, batch_size: int = 16, gamma: float = 0.99, loss:nn.modules.loss._Loss = nn.MSELoss(), target_update_interval: int = 100) -> None:
        self.q_main = CustomModel(state_dim=state_dim, action_dim=action_dim)
        self.q_target = CustomModel(state_dim=state_dim, action_dim=action_dim)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss = loss
        self.optimizer = torch.optim.Adam(self.q_main.parameters(), lr= 1e-2)
        self.target_update_interval = target_update_interval
        self.i = 0

    def step(self, trajectory: list):
        if self.i % self.target_update_interval == 0:
            self.q_target.load_state_dict(self.q_main.state_dict())

        self.replay_buffer.append(trajectory)

        if len(self.replay_buffer) < self.replay_buffer.maxlen:
            return

        batch = self.sample_from_replay()

        batched_states = np.stack(batch[:, 0])
        batched_actions = np.stack(batch[:, 1])
        batched_rewards = np.stack(batch[:, 2])
        batched_next_states = np.stack(batch[:, 3])

        q_all_actions_next_state = self.q_target(torch.from_numpy(batched_next_states).float()).detach().numpy()
        # Get max q for each row
        q_max_next_state = q_all_actions_next_state.max(axis=1)
 

        done_mask = (batch[:, 4].astype(float) * -1) + 1

        y = batched_rewards + self.gamma * q_max_next_state * done_mask

        self.optimizer.zero_grad()
        q_all_actions_current = self.q_main(torch.from_numpy(batched_states).float())
        # Get q value for actual selected action (batched_actions)
        q_current = q_all_actions_current[np.arange(self.batch_size), batched_actions]

        loss = self.loss(q_current, torch.from_numpy(y).float())
        loss.backward()
        self.optimizer.step()

        self.i += 1

    def sample_from_replay(self):
        batch_ids = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in batch_ids]
        return np.array(batch, dtype=object)

    def inference(self, x):
        with torch.no_grad():
            return self.q_main(x)