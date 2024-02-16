import torch

from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from e3nn import o3

class YMIR(nn.Module):
    
    def __init__(self, 
                 lmax: int = 3,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, 
                         **kwargs)
        sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        
    def forward(
        self,
        batch: Batch) -> torch.Tensor:
        
        x = batch.x
        pos = batch.pos
        
        edge_index = radius_graph(pos, r=self.r_max, batch=batch.batch)
        i, j = edge_index

        # Embeddings
        node_feats = self.node_embedding(x)
        
        vectors = pos[i] - pos[j]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        
        sh = self.spherical_harmonics(vectors)