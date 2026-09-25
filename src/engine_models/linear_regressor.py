import torch.nn as nn
import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.settings import DEVICE

class LinearReg(nn.Module):
    def __init__(self, input_dim=60, device = DEVICE):
        super(LinearReg, self).__init__()
        self.linear = nn.Linear(input_dim, 1, device=device)

    def forward(self,x):
        return self.linear(x)