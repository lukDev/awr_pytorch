import torch
from torch import nn


class Model(nn.Module):

    init_func = None

    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
        self.train()

        return y
