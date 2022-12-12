import numpy as np
from q.dqn import Dqn
import torch
import random

def main():
    dqn = Dqn(state_dim=4, action_dim=2, replay_buffer_size=1000, batch_size=4)

    t_s = [
        [
            np.array([0.0, 0.0, 1.0, 0.0]), # state
            0, # action
            -10, # reward
            np.array([0.0, 1.0, 0.0, 0.0]), # next state
            False, # done
        ],
        [
            np.array([0.0, 1.0, 0.0, 0.0]), # state
            0, # action
            150, # reward
            np.array([1.0, 0.0, 0.0, 0.0]), # next state
            True, # done
        ],
        [
            np.array([0.0, 0.0, 1.0, 0.0]), # state
            1, # action
            100, # reward
            np.array([0.0, 0.0, 0.0, 1.0]), # next state
            True, # done
        ],
        [
            np.array([0.0, 1.0, 0.0, 0.0]), # state
            1, # action
            -10, # reward
            np.array([0.0, 0.0, 1.0, 0.0]), # next state
            False, # done
        ]
    ]

    for i in range(100000):
        dqn.step(random.choice(t_s))
        if i % 1000 == 0:
            print(dqn.inference(torch.Tensor([0.0, 0.0, 1.0, 0.0])))
            print(dqn.inference(torch.Tensor([0.0, 1.0, 0.0, 0.0])))

if __name__ == '__main__':
    main()