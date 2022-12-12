import numpy as np
import torch.nn as nn
import torch

from collections import deque
from dqn_experience.model import CustomModel

class Dqn:
    def __init__(self, replay_buffer_size: int = 1000, batch_size: int = 16, gamma: float = 0.99, loss:nn.modules.loss._Loss = nn.MSELoss(), target_update_interval: int = 100, lr: float = 1e-4, create_model = lambda: CustomModel(state_dim=16, action_dim=4)) -> None:
        self.q_main = create_model()
        self.q_target = create_model()
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss = loss
        self.optimizer = torch.optim.Adam(self.q_main.parameters(), lr=lr)
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

        done_mask = (batch[:, 4].astype(float) * -1) + 1

        self.optimizer.zero_grad()

        q_all_actions_current = self.q_main(torch.from_numpy(batched_states).float())
        q_current = q_all_actions_current[np.arange(self.batch_size), batched_actions]

        q_max_next_state = q_all_actions_next_state[np.arange(self.batch_size), q_all_actions_current.argmax(axis=1)]
        y = batched_rewards + self.gamma * q_max_next_state * done_mask

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