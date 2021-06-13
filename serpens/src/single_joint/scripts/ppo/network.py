import torch
import torch.nn as nn


class MlpPolicy(nn.Module):
    def __init__(self, action_size, input_size=8):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 24).double()
        self.fc2 = nn.Linear(24, 24).double()
        self.fc3_pi = nn.Linear(24, self.action_size).double()
        self.fc3_v = nn.Linear(24, 1).double()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def pi(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_pi(x)
        return self.softmax(x)

    def v(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_v(x)
        return x