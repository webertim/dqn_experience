from torch import nn

class CustomModel(nn.Module):

    def __init__(self, state_dim: int = 8, action_dim: int = 5) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 16)
        self.l5 = nn.Linear(16, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.relu(self.l3(x2))
        x4 = self.relu(self.l4(x3))
        return self.l5(x4)