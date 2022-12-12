import numpy as np
from q.dqn import Dqn
import torch
import random

def main():
    dqn = Dqn(state_dim=8, action_dim=5, replay_buffer_size=1000, batch_size=16, lr=1e-4)

    for i in range(100000):
        dqn.step(random.choice(t_s))

if __name__ == '__main__':
    main()