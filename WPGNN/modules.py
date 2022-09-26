import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import SumAggregation


class EdgeModel(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_sizes=None, output_activation=False, 
                       w_init=None, b_init=None, name=None):
        super(EdgeModel, self).__init__()
        self.name = name
        self.input_size, self.output_size = input_size, output_size
        self.output_activation = output_activation

        layer_sizes = [output_size] if layer_sizes is None else layer_sizes
        self.n_layers = len(layer_sizes)

        self.layers = []
        for i in range(self.n_layers):
            layer_input_size = self.input_size if i == 0 else layer_sizes[i-1]
            newLinLayer = Lin(layer_input_size, output_size ,bias=True)
            
            #add Module to children list sucht that it will be recursivly found in self.apply (called in WPGNN.__init__)
            self.add_module(name='linear{0:03d}'.format(i), module=newLinLayer)
            self.layers.append(newLinLayer)
        layer_input_size = self.input_size if self.n_layers == 0 else layer_sizes[-1]
        newLinLayer = Lin(layer_input_size, output_size ,bias=True)
        self.add_module(name='linear_out', module=newLinLayer)
        self.layers.append(newLinLayer)
        #TODO initialize weights and bias non random


    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        #x = torch.cat([src, dest, edge_attr, u[batch]], 1)
        #TODO check on how to incorperate the batch here
        #u = torch.as_tensor([u for i in range(src.shape[0])], dtype=torch.float32)
        
        if batch is None:
            u = u.repeat(src.shape[0],1)
        else:
            u = u[batch.edge_attr_batch]
        x = torch.cat([edge_attr, dest,src, u], 1)


        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.layers[-1](x)
        if self.output_activation == 'leaky_relu':
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        elif self.output_activation == 'relu':
            x = torch.nn.functional.relu(x)
        elif self.output_activation == 'softplus':
            x = torch.nn.functional.softplus(x)
        elif self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'none':
            pass
        else:
            assert self.output_activation == False

        return x

class NodeModel(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_sizes=None, output_activation=False, 
                       w_init=None, b_init=None, name=None):
        super(NodeModel, self).__init__()
        self.name = name
        self.input_size, self.output_size = input_size, output_size
        self.output_activation = output_activation

        layer_sizes = [output_size] if layer_sizes is None else layer_sizes
        self.n_layers = len(layer_sizes)

        self.layers = []
        for i in range(self.n_layers):
            layer_input_size = self.input_size if i == 0 else layer_sizes[i-1]
            newLinLayer = Lin(layer_input_size, output_size ,bias=True)
            self.add_module(name='linear{0:03d}'.format(i), module=newLinLayer)
            self.layers.append(newLinLayer)
        layer_input_size = self.input_size if self.n_layers == 0 else layer_sizes[-1]
        newLinLayer = Lin(layer_input_size, output_size ,bias=True)
        self.add_module(name='linear_out', module=newLinLayer)
        self.layers.append(newLinLayer)

        self.aggregator = SumAggregation()
        #TODO initialize weights and bias non random


    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index

        #aggregate
        row = row
        col = col
        src_to_target_aggregated = self.aggregator(edge_attr,row)
        target_to_src_aggregated = self.aggregator(edge_attr,col)

        #pad to right dimensions as the output of the aggregator is equal to the biggest value of a src/target node
        dim_diff_src = x.shape[0] - src_to_target_aggregated.shape[0]
        dim_diff_target = x.shape[0] - target_to_src_aggregated.shape[0]

        src_to_target_aggregated = torch.nn.functional.pad(src_to_target_aggregated, (0,0,0,dim_diff_src))
        target_to_src_aggregated = torch.nn.functional.pad(target_to_src_aggregated, (0,0,0,dim_diff_target))

        #TODO replicate the global information u accoring to batch
        #u = torch.as_tensor([u for i in range(x.shape[0])], dtype=torch.float32)
        
        if batch is None:
            u = u.repeat(x.shape[0],1)
        else:
            u = u[batch.x_batch]
        tmp = torch.cat([target_to_src_aggregated, src_to_target_aggregated, x, u], dim=1)
        
        for layer in self.layers[:-1]:
            tmp = layer(tmp)
            tmp = torch.nn.functional.leaky_relu(tmp, negative_slope=0.2)

        tmp = self.layers[-1](tmp)
        if self.output_activation == 'leaky_relu':
            tmp = torch.nn.functional.leaky_relu(tmp, negative_slope=0.2)
        elif self.output_activation == 'relu':
            tmp = torch.nn.functional.relu(tmp)
        elif self.output_activation == 'softplus':
            tmp = torch.nn.functional.softplus(tmp)
        elif self.output_activation == 'sigmoid':
            tmp = torch.sigmoid(tmp)
        elif self.output_activation == 'none':
            pass
        else:
            assert self.output_activation == False

        return tmp


class GlobalModel(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_sizes=None, output_activation=False, 
                       w_init=None, b_init=None, name=None):
        super(GlobalModel, self).__init__()
        self.name = name
        self.input_size, self.output_size = input_size, output_size
        self.output_activation = output_activation

        layer_sizes = [output_size] if layer_sizes is None else layer_sizes
        self.n_layers = len(layer_sizes)

        self.layers = []
        for i in range(self.n_layers):
            layer_input_size = self.input_size if i == 0 else layer_sizes[i-1]
            newLinLayer = Lin(layer_input_size, output_size ,bias=True)
            self.add_module(name='linear{0:03d}'.format(i), module=newLinLayer)
            self.layers.append(newLinLayer)
        layer_input_size = self.input_size if self.n_layers == 0 else layer_sizes[-1]
        newLinLayer = Lin(layer_input_size, output_size ,bias=True)
        self.add_module(name='linear_out', module=newLinLayer)
        self.layers.append(newLinLayer)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        #TODO check in which dim we have to add/cat and wheter we need some batch here
        if batch is None:
            nodes_aggregated = x.sum(0)
            edges_aggregated = edge_attr.sum(0)
        else:
            z = torch.zeros(batch.x_batch[-1]+1, x.shape[1])
            nodes_aggregated = z.index_add_(0, batch.x_batch, x)
            edges_aggregated = torch.zeros(batch.x_batch[-1]+1, edge_attr.shape[1]).index_add_(0, batch.edge_attr_batch, edge_attr)

        tmp = torch.cat([edges_aggregated, nodes_aggregated, u],1)
        
        for layer in self.layers[:-1]:
            tmp = layer(tmp)
            tmp = torch.nn.functional.leaky_relu(tmp, negative_slope=0.2)

        tmp = self.layers[-1](tmp)
        if self.output_activation == 'leaky_relu':
            tmp = torch.nn.functional.leaky_relu(tmp, negative_slope=0.2)
        elif self.output_activation == 'relu':
            tmp = torch.nn.functional.relu(tmp)
        elif self.output_activation == 'softplus':
            tmp = torch.nn.functional.softplus(tmp)
        elif self.output_activation == 'sigmoid':
            tmp = torch.sigmoid(tmp)
        elif self.output_activation == 'none':
            pass
        else:
            assert self.output_activation == False

        return tmp

