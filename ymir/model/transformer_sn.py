import torch

from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn.math import (soft_one_hot_linspace,
                       soft_unit_step)
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from torch_geometric.data import Batch
from ymir.params import MAX_RADIUS

L_MAX = 3
NUMBER_OF_BASIS = 10
MOL_ID_EMBEDDING_SIZE = 16
NODE_Z_EMBEDDING_SIZE = 64 - MOL_ID_EMBEDDING_SIZE
MIDDLE_LAYER_SIZE = 32
MAX_NUMBER_TYPES = 100

class TransformerSN(torch.nn.Module):
    
    def __init__(self, 
                 irreps_output: o3.Irreps,
                 node_z_embedding_size: int = NODE_Z_EMBEDDING_SIZE,
                 node_mol_embedding_size: int = MOL_ID_EMBEDDING_SIZE,
                 lmax: int = L_MAX,
                 max_radius: float = MAX_RADIUS,
                 number_of_basis: int = NUMBER_OF_BASIS,
                 middle_layer_size: int = MIDDLE_LAYER_SIZE,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_z_embedding_size = node_z_embedding_size
        self.node_mol_embedding_size = node_mol_embedding_size
        self.irreps_output = irreps_output
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.middle_layer_size = middle_layer_size
        
        self.node_z_embedder = torch.nn.Embedding(num_embeddings=MAX_NUMBER_TYPES, 
                                                  embedding_dim=self.node_z_embedding_size)
        self.node_mol_embedder = torch.nn.Embedding(num_embeddings=3, # seed, pocket or fragment
                                                    embedding_dim=self.node_mol_embedding_size)
        
        self.node_embedding_size = self.node_z_embedding_size + self.node_mol_embedding_size
        self.irreps_input_node = o3.Irreps(f'{self.node_embedding_size}x0e')
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, 
                                        normalize=True, 
                                        normalization="component")
        
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
    
    def forward(self,
                batch: Batch):
        x = batch.x
        pos = batch.pos
        mol_id = batch.mol_id
        
        x = self.node_z_embedder(x)
        mol_id = self.node_mol_embedder(mol_id)
        
        x = torch.cat([x, mol_id], dim=-1)

        edge_vec = pos
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
        
        value_weights = self.value_weights_fc(edge_length_embedded)
        value = self.tp_value(x=x, 
                              y=edge_sh, 
                              weight=value_weights)
        
        output = scatter(src=value,
                         index=batch.batch, 
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
    