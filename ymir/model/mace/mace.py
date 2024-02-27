###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import (Any, 
                    Callable,
                    Dict, 
                    List, 
                    Optional, 
                    Type, 
                    Union)

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from torch_scatter import scatter_mean

from mace.modules.blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
    RealAgnosticResidualInteractionBlock
)

from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import unbatch

# pylint: disable=C0302

R_MAX_MACE = 5.0
AVG_NUM_NEIGHBORS = 2 * R_MAX_MACE

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        num_elements: int,
        r_max: float = R_MAX_MACE,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 3,
        interaction_cls: Type[InteractionBlock] = RealAgnosticResidualInteractionBlock,
        interaction_cls_first: Type[InteractionBlock] = RealAgnosticResidualInteractionBlock,
        num_interactions: int = 2,
        avg_num_neighbors: float = AVG_NUM_NEIGHBORS,
        correlation: Union[int, List[int]] = 3,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        # radius_cutoff: float = 5.0,
    ):
        super().__init__()
        
        # self.radius_cutoff = radius_cutoff
        # self.r_max = r_max
        
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        inter: InteractionBlock = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        for i in range(num_interactions - 1):
            hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)

    def forward(
        self,
        batch: Batch) -> torch.Tensor:

        x = batch.x 
        pos = batch.pos
        
        # ohe_x = torch.nn.functional.one_hot(x, num_classes=70) #70 is large enough for atomic num up to 69

        edge_index = radius_graph(pos, r=self.r_max, batch=batch.batch)
        j, i = edge_index

        # import pdb;pdb.set_trace()

        # Embeddings
        node_feats = self.node_embedding(x)
        
        vectors = pos[j] - pos[i]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True) # might need to implement shift
        
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        # import pdb;pdb.set_trace()

        # Interactions
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=x,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=x,
            )

        if torch.isnan(node_feats).any():
            import pdb;pdb.set_trace()

        env_vector = scatter_mean(src=node_feats, 
                                  index=batch.batch, 
                                  dim=0)
        
        # is_attach_point = x[:, 0] == 1
        # env_vector = node_feats[is_attach_point]
        
        # n_pockets = batch.batch.max() + 1
        # assert env_vector.shape[0] == n_pockets

        # unbatched_contributions = unbatch(node_feats, batch=batch.batch)
        
        # env_list = []
        # for contrib in unbatched_contributions:
        #     env_list.append(contrib[-1]) # we assume the new frag environement is the last one per batch
        #     # might need to be modified to fit the Aggregation paradigm in torch geometric
        
        # env_vector = torch.stack(env_list)
        
        return env_vector

        return node_feats_out