import torch

from torch import nn
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn.math import (soft_one_hot_linspace,
                       soft_unit_step)
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from torch_geometric.data import Batch
from ymir.params import CNN_RADIUS, N_INTERACTION_BLOCKS

L_MAX = 3
NUMBER_OF_BASIS = 10
MOL_ID_EMBEDDING_SIZE = 16
NODE_Z_EMBEDDING_SIZE = 64 - MOL_ID_EMBEDDING_SIZE
MIDDLE_LAYER_SIZE = 32
MAX_NUMBER_TYPES = 100

class InteractionBlock(torch.nn.Module):
    
    def __init__(self, 
                 irreps_input_node: o3.Irreps,
                 irreps_sh: o3.Irreps,
                 irreps_output: o3.Irreps,
                 number_of_basis: int = NUMBER_OF_BASIS,
                 middle_layer_size: int = MIDDLE_LAYER_SIZE,) -> None:
        super().__init__()
        self.irreps_input_node = irreps_input_node
        self.irreps_sh = irreps_sh
        self.irreps_output = irreps_output
        self.number_of_basis = number_of_basis
        self.middle_layer_size = middle_layer_size
        
        # tp = Tensor Product
        self.tp_value = o3.FullyConnectedTensorProduct(irreps_in1=self.irreps_input_node, 
                                                        irreps_in2=self.irreps_sh, 
                                                        irreps_out=self.irreps_output, 
                                                        shared_weights=False)
        
        # hs has the form [input layer, middle layer, output layer]
        self.value_weights_hs = [self.number_of_basis, 
                                 self.middle_layer_size, 
                                 self.tp_value.weight_numel]
        self.value_weights_fc = FullyConnectedNet(hs=self.value_weights_hs, 
                                                  act=torch.nn.functional.silu)
        
        self.layer_norm = nn.LayerNorm(self.irreps_output.dim)
        # self.layer_norm = nn.BatchNorm1d(self.irreps_output.dim)
        
    def forward(self,
                x: torch.Tensor, # (num_nodes, feature_size)
                edge_sh: torch.Tensor, # (num_edges, sh_dim)
                edge_length_embedded: torch.Tensor, # (num_edges, num_basis),
                edge_src: torch.Tensor, # (num_edges)
                edge_dst: torch.Tensor, # (num_edges)
                avg_num_neighbors: float,
                ):
        
        value_weights = self.value_weights_fc(edge_length_embedded)
        edge_values = self.tp_value(x=x[edge_src], 
                              y=edge_sh, 
                              weight=value_weights)
        
        node_values = scatter(edge_values, 
                              edge_dst, 
                              dim=0)
        node_values = node_values / (avg_num_neighbors ** 0.5)
        node_values = self.layer_norm(node_values)
        return node_values


class CNN(torch.nn.Module):
    
    def __init__(self, 
                 hidden_irreps: o3.Irreps,
                 irreps_output: o3.Irreps,
                 num_elements: int,
                 node_z_embedding_size: int = NODE_Z_EMBEDDING_SIZE,
                 node_mol_embedding_size: int = MOL_ID_EMBEDDING_SIZE,
                 lmax: int = L_MAX,
                 max_radius: float = CNN_RADIUS,
                 number_of_basis: int = NUMBER_OF_BASIS,
                 middle_layer_size: int = MIDDLE_LAYER_SIZE,
                 n_interaction_blocks: int = N_INTERACTION_BLOCKS,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_irreps = hidden_irreps
        self.irreps_output = irreps_output
        self.num_elements = num_elements
        self.node_z_embedding_size = node_z_embedding_size
        self.node_mol_embedding_size = node_mol_embedding_size
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.middle_layer_size = middle_layer_size
        self.n_interaction_blocks = n_interaction_blocks
        
        self.node_z_embedder = torch.nn.Embedding(num_embeddings=self.num_elements, 
                                                  embedding_dim=self.node_z_embedding_size)
        self.node_mol_embedder = torch.nn.Embedding(num_embeddings=2, # seed, pocket or fragment
                                                    embedding_dim=self.node_mol_embedding_size)
        
        self.node_embedding_size = self.node_z_embedding_size + self.node_mol_embedding_size
        self.irreps_input_node = o3.Irreps(f'{self.node_embedding_size}x0e')
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, 
                                        normalize=True, 
                                        normalization="component")
        
        interaction_blocks: list[InteractionBlock] = []
        for i in range(self.n_interaction_blocks):
            if i == 0 :
                irreps_input_node = self.irreps_input_node
            else: # after the first IB, we are in equivariant features
                irreps_input_node = self.hidden_irreps
            if i == self.n_interaction_blocks - 1:
                irreps_output = self.irreps_output
            else:
                irreps_output = self.hidden_irreps
            interaction_block = InteractionBlock(irreps_input_node=irreps_input_node,
                                                irreps_sh=self.irreps_sh,
                                                irreps_output=irreps_output,
                                                number_of_basis=self.number_of_basis,
                                                middle_layer_size=self.middle_layer_size)
            interaction_blocks.append(interaction_block)
            
        self.interaction_blocks = torch.nn.ModuleList(interaction_blocks)
        
    
    def forward(self,
                batch: Batch):
        x = batch.x
        pos = batch.pos
        mol_id = batch.mol_id
        
        edge_src, edge_dst = radius_graph(pos, 
                                          self.max_radius,
                                          batch=batch.batch)
        
        # import pdb;pdb.set_trace()
        
        num_nodes = len(batch.x)
        avg_num_neighbors = len(edge_src) / num_nodes
        
        # import pdb;pdb.set_trace()
        
        x = self.node_z_embedder(x)
        mol_id = self.node_mol_embedder(mol_id)
        
        x = torch.cat([x, mol_id], dim=-1)

        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)

        edge_sh = self.sh(edge_vec)
        
        for interaction_block in self.interaction_blocks:
            x = interaction_block.forward(x,
                                          edge_sh,
                                          edge_length_embedded,
                                          edge_src,
                                          edge_dst,
                                          avg_num_neighbors)
            
        return x
            
        output = scatter(x,
                         batch.batch,
                         dim=0,
                         reduce='mean')
        
        # rot = o3.rand_matrix().to(pos)
        # pos_r = batch.pos @ rot.T

        # edge_vec_r = pos_r
        # edge_length_r = edge_vec_r.norm(dim=1)

        # edge_length_embedded_r = soft_one_hot_linspace(
        #     x=edge_length_r,
        #     start=0.0,
        #     end=self.max_radius,
        #     number=self.number_of_basis,
        #     basis='smooth_finite',
        #     cutoff=True
        # )
        # edge_length_embedded_r = edge_length_embedded_r.mul(self.number_of_basis**0.5)

        # edge_sh_r = self.sh(edge_vec_r)
        
        # value_weights_r = self.value_weights_fc(edge_length_embedded_r)
        # value_r = self.tp_value(x=x, 
        #                       y=edge_sh_r, 
        #                       weight=value_weights_r)

        # output_r = scatter(src=value_r,
        #                  index=batch.batch, 
        #                  dim=0,
        #                  reduce='mean')

        return output
    