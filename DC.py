import torch
import torch.nn.functional as F
from torch import nn

class DCModel(nn.Module):
    def __init__(
        self,
        N_turbines: int,
        N_hidden_layers: int,
        hidden_dim: int = 64,
        include_ws: bool = True,
        include_wd: bool = False,
        include_dropout: bool = False,
        **kwargs
    ):
        super().__init__()
        if not (include_ws or include_wd or include_dropout):
            raise RuntimeError("When initializing DCModel, \
                at least one of include_ws, include_wd, and include_dropout must be True.")
        input_size = 0
        if include_ws:
            input_size += 1 # ws
        if include_wd:
            input_size += 2 # sin(wd), cos(wd)
        if include_dropout:
            input_size += N_turbines # turbine status (T/F for each turbine)
            
        self.input_layer = nn.Linear(input_size, hidden_dim)
        
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_1=nn.BatchNorm1d(hidden_dim)

        self.N_hidden_layers = N_hidden_layers
        if N_hidden_layers >= 2:
            self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn_2=nn.BatchNorm1d(hidden_dim)
        if N_hidden_layers >= 3:
            self.hidden_layer_3 = nn.Linear(hidden_dim, hidden_dim)
            self.bn_3=nn.BatchNorm1d(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, N_turbines*2)
        
    def forward(self, x):
        # x: state of system e.g. torch.tensor([ws, sin(wd), cos(wd), status T/F for each turbine])
        # u: control action (torch.tensor([yaw angle for each turbine]))
        u = F.relu(self.input_layer(x)) # input layer
        
        u = self.hidden_layer_1(u) # hidden layer
        u = self.bn_1(u) # batch norm
        u = F.relu(u) # activation function
        
        if self.N_hidden_layers >= 2:
            u = self.hidden_layer_2(u) # hidden layer
            u = self.bn_2(u)
            u = F.relu(u) # activation layer
        
        if self.N_hidden_layers >= 3:
            u = self.hidden_layer_3(u) # hidden layer
            u = self.bn_3(u) # batch norm
            u = F.relu(u) # AF
        
        u = torch.sigmoid(self.output_layer(u))-.5 # output layer
        return u