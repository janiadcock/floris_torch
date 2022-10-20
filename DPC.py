import torch
import torch.nn.functional as F
from torch import nn

class DPCModel(nn.Module):
    def __init__(
        self,
        n_turbines: int,
        hidden_dim: int = 64,
        dropout: bool = False,
        **kwargs
    ):
        super().__init__()
        input_size = 2 # ws, wd
        if dropout:
            input_size += n_turbines # dropout (T/F for each turbine)
        self.input_layer = nn.Linear(input_size, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_turbines)
        
    def forward(self, x):
        # x: state of system = torch.tensor([ws, wd])
        # u: control action (yaw angle for each turbine)
        u = F.relu(self.input_layer(x))
        u = F.relu(self.hidden_layer(u))
        u = torch.sigmoid(self.output_layer(u))
        return u