import torch
from torch import nn

from models.model import Model
from utilities.utils import device, xavier_init


class Critic(Model):

    name = "critic"

    def __init__(self):
        super(Critic, self).__init__()

        self.fc = None
        self.optimiser = None
        self.criterion = torch.nn.MSELoss()

    def set_params(self, hyper_ps):
        hidden_size = hyper_ps['c_hidden_size']
        hidden_layers = hyper_ps['c_hidden_layers']

        fcs = [
            nn.Linear(hyper_ps['state_dim'], hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        ]
        for _ in range(hidden_layers):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            fcs.append(nn.ReLU())
            fcs.append(nn.BatchNorm1d(hidden_size))
        fcs.append(nn.Linear(hidden_size, 1))

        self.fc = nn.Sequential(*fcs)

        self.optimiser = torch.optim.SGD(
            lr=hyper_ps['c_learning_rate'],
            momentum=hyper_ps['c_momentum'],
            params=self.parameters()
        )

    def forward(self, state):
        return self.fc(state)

    def backward(self, out, target):
        loss = self.criterion(out, target.float())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss
