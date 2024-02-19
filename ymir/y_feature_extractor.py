import torch

from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn.math import (soft_one_hot_linspace,
                       soft_unit_step)
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from torch_geometric.data import Batch
from ymir.params import MAX_RADIUS
from ymir.atomic_num_table import AtomicNumberTable

L_MAX = 3
NUMBER_OF_BASIS = 10
MOL_ID_EMBEDDING_SIZE = 16
NODE_Z_EMBEDDING_SIZE = 64 - MOL_ID_EMBEDDING_SIZE
MIDDLE_LAYER_SIZE = 32
MAX_NUMBER_TYPES = 100

class YFeatureExtractor(torch.nn.Module):
    
    def __init__(self, 
                 atomic_num_table: AtomicNumberTable,
                 lmax: int = L_MAX,
                 max_radius: float = MAX_RADIUS,
                 number_of_basis: int = NUMBER_OF_BASIS,
                 summing_embeddings: bool = False) -> None:
        super().__init__()
        self.z_table = atomic_num_table
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.summing_embeddings = summing_embeddings
        
        self.irreps_z = o3.Irreps(f'{len(self.z_table)}x0e')
        self.irreps_basis = o3.Irreps(f'{self.number_of_basis}x0e')
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, 
                                        normalize=True, 
                                        normalization="component")
        
        self.element_shell_tp = o3.FullTensorProduct(irreps_in1=self.irreps_z, 
                                                  irreps_in2=self.irreps_sh)
            
        self.element_dist_shell_tp = o3.FullTensorProduct(irreps_in1=self.element_shell_tp.irreps_out, 
                                                       irreps_in2=self.irreps_basis)
            
        self.irreps_out = self.element_dist_shell_tp.irreps_out
    
    def forward(self,
                batch: Batch):
        
        x = batch.x # one hot encoding of the atomic number (num_nodes, num_elements)
        pos = batch.pos # index of the original graph (num_nodes)

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

        element_shell = self.element_shell_tp(x, edge_sh)
        
        element_dist_shell = self.element_dist_shell_tp(element_shell, edge_length_embedded)
        
        if self.summing_embeddings:
            reduce = 'sum'
        else:
            reduce = 'mean'
        
        output = scatter(src=element_dist_shell,
                         index=batch.batch, 
                         dim=0,
                         reduce=reduce)

        return output
    