import torch
from meta import MetaLayer
import modules as mod
import numpy as np 
from torch_geometric.loader import DataLoader
from time import time
import utils
from torch_geometric.nn import SumAggregation
import h5py
import os





class WPGNN(torch.nn.Module):
    '''
        Parameters:
            eN_in, eN_out   - number of input/output edge features
            nN_in, nN_out   - number of input/output node features
            gN_in, gN_out   - number of input/output graph features
            n_layers        - number of graph layers in the network
            graph_layers    - list of graph layers
            model_path      - location of a saved model, if None then use randomly initialized weights
            scale_factors   - list of scaling factors used to normalize data
            optmizer        - Sonnet optimizer object that will be used for training
    '''
    def __init__(self, eN=2, nN=3, gN=3, graph_size=None,
                       scale_factors=None, model_path=None, name=None, h5=True):
        super(WPGNN, self).__init__()

        # Set model architecture
        self.eN_in,  self.nN_in,  self.gN_in  = eN, nN, gN
        if graph_size is None:
            graph_size = [[32, 32, 32],
                          [16, 16, 16],
                          [16, 16, 16],
                          [ 8,  8,  8],
                          [ 8,  8,  8],
                          [ 4,  2,  2]]
        self.n_layers = len(graph_size)
        self.eN_out, self.nN_out, self.gN_out = graph_size[-1][0], graph_size[-1][1], graph_size[-1][2]

        # Construct WPGNN model
        self.graph_layers = []
        for i in range(self.n_layers - 1):
            dim_in = [self.eN_in, self.nN_in, self.gN_in] if i == 0 else graph_size[i-1]
            newMetaLayer = self.graph_layer(dim_in, graph_size[i],
                                                      n_layers=2,
                                                      output_activation='sigmoid',
                                                      layer_index=i)
            #add Module to children list such that it will be recursivly found in self.apply
            self.add_module(name='meta{0:03d}'.format(i) ,module=newMetaLayer)
            self.graph_layers.append(newMetaLayer)
        newMetaLayer = self.graph_layer(graph_size[-2], graph_size[-1],
                                                  n_layers=1,
                                                  output_activation='relu',
                                                  layer_index=i+1)
        self.add_module(name='meta{0:03d}'.format(self.n_layers - 1), module=newMetaLayer)
        self.graph_layers.append(newMetaLayer)

        if scale_factors is None:
            self.scale_factors = {'x_globals': np.array([[0., 25.], [0., 25.], [0.09, 0.03]]),
                                    'x_nodes': np.array([[0., 75000.], [0., 85000.], [15., 15.]]),
                                    'x_edges': np.array([[-100000., 100000.], [0., 75000.]]),
                                  'f_globals': np.array([[0., 500000000.], [0., 100000.]]),
                                    'f_nodes': np.array([[0., 5000000.], [0.,25.]]),
                                    'f_edges': np.array([[0., 0.]])}
        else:
            self.scale_factors = scale_factors

        #init weights
        self.apply(init_weights) 

        if model_path is not None:
            if h5:
                load_weights_h5(self,model_path)
            else:
                self.custom_load_weights(model_path)
        


        self.optimizer = torch.optim.Adam(self.parameters())


    def forward(self, x, edge_index, edge_attr, u, batch=None): #, physical_units=False): what is that doing 
        # Evaluate the WPGNN on a given input graph
        for graph_layer in self.graph_layers:

            x_out, edge_attr_out, u_out = graph_layer.forward(x, edge_index, edge_attr, u, batch)
            

            #skip connections
            tf_edge_dims = (edge_attr.shape[1] == edge_attr_out.shape[1])
            tf_node_dims = (x.shape[1] == x_out.shape[1])
            tf_global_dims = (u.shape[0] == u_out.shape[0])
            if tf_edge_dims & tf_node_dims & tf_global_dims :
                x_out = torch.add(x_out, x)
                edge_attr_out = torch.add(edge_attr_out, edge_attr)
                u_out = torch.add(u_out, u)
            
            x = x_out
            edge_attr = edge_attr_out
            u = u_out
        
        return x, edge_attr, u


    def graph_layer(self, dim_in, dim_out, n_layers=3, output_activation='relu', layer_index=0):
        edge_inputs, edge_outputs = dim_in[0] + 2*dim_in[1] + dim_in[2], dim_out[0]
        layer_sizes = [edge_outputs for _ in range(n_layers-1)]
        eModel = mod.EdgeModel(edge_inputs, edge_outputs, layer_sizes=layer_sizes,
                                     output_activation=output_activation,
                                     name='edgeUpdate{0:02d}'.format(layer_index))
        
        #use already processed edge_attr from EdgeModel as input of NodeModel
        node_inputs, node_outputs = 2*dim_out[0] + dim_in[1] + dim_in[2], dim_out[1]
        layer_sizes = [node_outputs for _ in range(n_layers-1)]
        nModel = mod.NodeModel(node_inputs, node_outputs, layer_sizes=layer_sizes,
                                     output_activation=output_activation,
                                     name='nodeUpdate{0:02d}'.format(layer_index))

        #use already processed edge_attr, x from EdgeModel as input of GlobalModel
        global_inputs, global_outputs = dim_out[0] + dim_out[1] + dim_in[2], dim_out[2]
        layer_sizes = [global_outputs for _ in range(n_layers-1)]
        gModel = mod.GlobalModel(global_inputs, global_outputs, layer_sizes=layer_sizes,
                                         output_activation=output_activation,
                                         name='globalUpdate{0:02d}'.format(layer_index))

        return MetaLayer(eModel, nModel, gModel)
    
    def custom_save_Model(self, filename):
        torch.save(self.state_dict(), filename)



    def custom_load_weights(self, filename):
        #SIMPLIFIED
        self.load_state_dict(torch.load(filename))

    
    def compute_dataset_loss(self, data, batch_size=100, reporting=False):
        # Compute the mean loss across an entire data without updating the model parameters
        dataset, f, u = utils.tfData_to_pygData(data)
        loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=False)

        if reporting:
            N, l_tot, l_tp_tot, l_ts_tot, l_pp_tot, l_ps_tot = 0., 0., 0., 0., 0., 0.
        else:
            N, l_tot = 0., 0.
            
        batch_iterator = iter(loader)
        for batch in batch_iterator:
            N_batch = len(batch)

            x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
            l = self.compute_loss(x_out, u_out, f, batch, reporting=True)

            if reporting:
                l_tot += l[0]*N_batch
                l_tp_tot += l[1]*N_batch
                l_ts_tot += l[2]*N_batch
                l_pp_tot += l[3]*N_batch
                l_ps_tot += l[4]*N_batch
            else:
                l_tot += l*N_batch
            N += N_batch

        if reporting:
            return l_tot/N, l_tp_tot/N, l_ts_tot/N, l_pp_tot/N, l_ps_tot/N
        else:
            return l_tot/N
    
    def compute_loss(self, x_out, u_out, f, batch, reporting=False):
        f_nodes = torch.cat(tuple(np.take(f[0],batch.y)))
        f_globals = torch.as_tensor(np.take(f[1],batch.y,axis=0), dtype=torch.float32)

        # Compute the mean squared error for the target turbine- and plant-level outputs
        turbine_loss = torch.mean((x_out - f_nodes)**2, axis=0)
        plant_loss = torch.mean((u_out - f_globals)**2, axis=0)
        
        loss = torch.sum(plant_loss) + 10.*torch.sum(turbine_loss)

        if reporting:
            return loss, turbine_loss[0], turbine_loss[1], plant_loss[0], plant_loss[1]
        else:
            return loss

    def train_step(self, batch, f, u):
        self.optimizer.zero_grad()
        x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
        loss = self.compute_loss(x_out, u_out, f, batch)
        loss.backward()
        self.optimizer.step()
        return loss
        
    def fit(self, train_data, test_data=None, batch_size=2, learning_rate=1e-3, decay_rate=0.99,
                  epochs=100, print_every=10, save_every=100, save_model_path=None, h5=True):
        '''
            Parameters:
                train_data       - training data in (list of input graphs, list of output graphs) format
                test_data        - test data used to monitor training progress, same format as training data
                batch_size       - number of samples to include in each training batch
                learning_rate    - learning rate for the training optimizer
                decay_rate       - rate of decay for the learning rate
                epochs           - the total number of epochs of training to perform
                print_every      - how frequently (in training iterations) to print the batch performance
                save_every       - how frequently (in epochs) to save the model
                save_model_path  - path to directory where to save model during training
        '''
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

        # Build data pipelines
        dataset, f, u = utils.tfData_to_pygData(train_data)
        loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=False)
        #loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=True)

        # Start training process
        iters = 0
        for epoch in range(1, epochs+1):
            start_time = time()
            print('Beginning epoch {}...'.format(epoch))

            batch_loss = 0
            batch_iterator = iter(loader)
            for idx_batch in batch_iterator:


                self.train_step(idx_batch, f, u)

                if (print_every > 0) and ((iters % print_every) == 0):
                    x_out, edge_attr_out, u_out  = self.forward(idx_batch.x, idx_batch.edge_index, idx_batch.edge_attr, u[idx_batch.y], idx_batch)
                    l = self.compute_loss(x_out, u_out, f, idx_batch, reporting=True)
                    print('Total batch loss = {:.6f}'.format(l[0]))
                    print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
                    print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
                    print('')

                iters += 1
            
            # Save current state of the model
            if (save_model_path is not None) and ((epoch % save_every) == 0):
                model_epoch = save_model_path+'/{0:05d}'.format(epoch)
                if not os.path.exists(model_epoch):
                    os.makedirs(model_epoch)
                if h5:
                    save_weights_h5(self,'/'.join([model_epoch, 'wpgnn.h5']))
                else:
                    self.custom_save_Model('/'.join([model_epoch, 'wpgnn.pt']))


            
            # Report current training/testing performance of model
            l = self.compute_dataset_loss(train_data, batch_size=batch_size, reporting=True)
            print('Epochs {} Complete'.format(epoch))
            print('Training Loss = {:.6f}, '.format(l[0]))
            print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
            print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
            
            if test_data is not None:
                l = self.compute_dataset_loss(test_data, batch_size=batch_size, reporting=True)
                print('Testing Loss = {:.6f}, '.format(l[0]))
                print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
                print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
            

            for g in self.optimizer.param_groups:
                g['lr'] *= decay_rate

            print('Time to complete: {0:02f}\n'.format(time() - start_time), flush=True)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight,1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def load_weights_h5(m, filename):
    with h5py.File(filename, 'r') as f:
        #m is wpgnn
        for index, childMeta in enumerate(m.children()):
            #childMeta is a MetaLayer
            for childModel in childMeta.children():
                #childeModel is Edge/Node/Global-Model
                for name, childLin in childModel.named_children():
                    if not isinstance(childLin, SumAggregation):
                        if isinstance(childModel, mod.EdgeModel):
                            weight_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/b:0'
                        if isinstance(childModel, mod.NodeModel):
                            weight_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/b:0'
                        if isinstance(childModel, mod.GlobalModel):
                            weight_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/b:0'
                        childLin.weight= torch.nn.Parameter(torch.tensor(f[weight_name][()].transpose(), dtype=torch.float32))
                        childLin.bias= torch.nn.Parameter(torch.tensor(f[bias_name][()], dtype=torch.float32))


def save_weights_h5(m, filename):
    with h5py.File(filename, 'w') as f:
        #m is wpgnn
        for index, childMeta in enumerate(m.children()):
            #childMeta is a MetaLayer
            for childModel in childMeta.children():
                #childeModel is Edge/Node/Global-Model
                for name, childLin in childModel.named_children():
                    if not isinstance(childLin, SumAggregation):
                        if isinstance(childModel, mod.EdgeModel):
                            weight_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/b:0'
                        if isinstance(childModel, mod.NodeModel):
                            weight_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/b:0'
                        if isinstance(childModel, mod.GlobalModel):
                            weight_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/b:0'
                        try:
                            f.create_group(weight_name)
                            f.create_group(bias_name)
                        except:
                            pass
                        f[weight_name].create_dataset(weight_name, data=childLin.weight.detach().numpy())
                        f[bias_name].create_dataset(bias_name, data=childLin.bias.detach().numpy())

