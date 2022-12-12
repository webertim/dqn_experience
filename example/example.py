import numpy as np
import torch
import random

from dqn_experience import Dqn
from dqn_experience import CustomModel

def main():
    dqn = Dqn(create_model=lambda: CustomModel(state_dim=4, action_dim=2))

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