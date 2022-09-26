from typing import Optional, Tuple
import torch
from torch import Tensor


class MetaLayer(torch.nn.Module):
    #Slight modification of torch_geometric.nn.meta
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
            self, x: Tensor, edge_index: Tensor,
            edge_attr: Optional[Tensor] = None, u: Optional[Tensor] = None,
            batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """"""
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            #This is the only line that is different to torch_geometric.nn.meta
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u, batch)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')
