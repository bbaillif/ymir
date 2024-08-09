import torch

from torch import nn
from torch_geometric.data import Batch
from e3nn import o3
from ymir.distribution import CategoricalMasked
from typing import NamedTuple
from ymir.params import (EMBED_HYDROGENS, 
                         L_MAX, 
                         HIDDEN_IRREPS,
                         IRREPS_OUTPUT,
                         TORSION_ANGLES_DEG,
                         COMENET_CONFIG)
from ymir.atomic_num_table import AtomicNumberTable
from ymir.data.fragment import Fragment
from torch_geometric.data import Batch
from ymir.model import CNN, ComENetModel, EGNN
from torch_scatter import scatter
from ymir.featurizer_sn import Featurizer
from torch_geometric.data import Data
from torch_cluster import radius_graph
    
class Action(NamedTuple): 
    frag_i: torch.Tensor # (batch_size)
    frag_logprob: torch.Tensor # (batch_size)
    frag_entropy: torch.Tensor # (batch_size)
    
    
class MultiLinear(nn.Module):
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list[int],
                 output_dim: int,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, 
                         **kwargs)
        assert len(hidden_dims) > 0
        dims = [input_dim] + hidden_dims
        modules = []
        # first_hidden = True
        for dim1, dim2 in zip(dims, dims[1:]):
            modules.append(nn.Linear(dim1, dim2))
            # if first_hidden:
                # modules.append(nn.Tanh())
                # first_hidden = False
            # else:
            modules.append(nn.SiLU())
            # modules.append(nn.Dropout(p=0.25))
        modules.append(nn.Linear(dims[-1], output_dim))
        # modules.append(nn.Tanh())

        self.sequential = nn.Sequential(*modules)
        
    def forward(self,
                x: torch.Tensor):
        return self.sequential(x)
    
    
class Aggregator(nn.Module):
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, 
                         **kwargs)
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        # self.layer_norm = nn.BatchNorm1d(output_dim)
    
    def forward(self,
                x: torch.Tensor,
                batch: torch.Tensor):
        
        x = self.linear(x)
        x = self.layer_norm(x)
        output = scatter(x,
                         batch,
                         dim=0,
                         reduce='mean')
        return output
    
    
class Agent(nn.Module):
    
    def __init__(self, 
                 protected_fragments: list[Fragment],
                 atomic_num_table: AtomicNumberTable,
                 features_dim: int,
                 hidden_irreps: o3.Irreps = HIDDEN_IRREPS,
                 irreps_output: o3.Irreps = IRREPS_OUTPUT,
                #  lmax: int = L_MAX,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 torsion_angle_deg: list[float] = TORSION_ANGLES_DEG,
                 pocket_feature_type: str = 'soap',
                 nn_type: str = 'egnn'
                 ):
        super(Agent, self).__init__()
        self.protected_fragments = protected_fragments
        self.z_table = atomic_num_table
        self.features_dim = features_dim
        self.hidden_irreps = hidden_irreps
        self.irreps_output = irreps_output
        # self.lmax = lmax
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        self.torsion_angles_deg = torsion_angle_deg
        self.pocket_feature_type = pocket_feature_type
        self.nn_type = nn_type
        
        self.n_fragments = len(self.protected_fragments)
        self.action_dim = self.n_fragments
        
        # self.pocket_irreps = o3.Irreps(f'128x0e')
        # n_hidden_features = self.pocket_irreps.dim
        n_hidden_features = 128
        # self.pocket_feature_extractor = CNN(hidden_irreps=o3.Irreps(f'16x0e + 16x1o + 16x2e'),
        #                                     irreps_output=self.pocket_irreps,
        #                                      num_elements=len(self.z_table))
        
        if self.pocket_feature_type == 'soap':
            self.pocket_feature_extractor = MultiLinear(self.features_dim, 
                                                        [self.features_dim, self.features_dim // 2], 
                                                        n_hidden_features)
        else: # 'graph'
            if nn_type == 'cnn':
                # self.pocket_feature_extractor = ComENetModel(config=COMENET_CONFIG,
                #                                             features_dim=n_hidden_features)
                self.pocket_irreps = o3.Irreps(f'{n_hidden_features}x0e')
                self.pocket_feature_extractor = CNN(hidden_irreps=o3.Irreps(f'64x0e + 16x1o + 16x2e'),
                                                    irreps_output=self.pocket_irreps,
                                                    num_elements=len(self.z_table))
            else:
                self.pocket_feature_extractor = EGNN(in_node_nf=64,
                                                        hidden_nf=64,
                                                        out_node_nf=n_hidden_features,
                                                        n_layers=3)
        
        # self.pocket_feature_extractor = MultiLinear(self.features_dim, [256, 128], n_hidden_features)
        
        self.actor = nn.Linear(n_hidden_features, self.n_fragments)
        self.critic = nn.Linear(n_hidden_features, 1)
        # self.actor = MultiLinear(n_hidden_features, [128, 64], self.n_fragments)
        # self.critic = MultiLinear(n_hidden_features, [128, 64], 1)
        
        self.num_elements = len(self.z_table)
        self.node_mol_embedding_size = 8
        self.focal_embedding_size = 16
        self.node_z_embedding_size = 64 - self.node_mol_embedding_size - self.focal_embedding_size
        
        self.node_z_embedder = torch.nn.Embedding(num_embeddings=self.num_elements, 
                                                  embedding_dim=self.node_z_embedding_size)
        self.node_mol_embedder = torch.nn.Embedding(num_embeddings=3, # seed, pocket or void
                                                    embedding_dim=self.node_mol_embedding_size)
        self.focal_embedder = torch.nn.Embedding(num_embeddings=2, # focal or not
                                                embedding_dim=self.focal_embedding_size)


    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        action = self.get_action(features, masks)
        value = self.get_value(features)
        return action, value
    
    
    def extract_features(self,
                         x: Batch):
        if self.pocket_feature_type == 'soap':
            features = self.pocket_feature_extractor(x)
        elif self.pocket_feature_type == 'graph':
            # features = self.pocket_feature_extractor(x)
            if self.nn_type == 'cnn':
                features = self.pocket_feature_extractor.get_atomic_contributions(x)
            else:
                batch = x
                x = batch.x
                pos = batch.pos
                mol_id = batch.mol_id
                is_focal = batch.is_focal
                
                x = self.node_z_embedder(x)
                mol_id = self.node_mol_embedder(mol_id)
                is_focal = self.focal_embedder(is_focal)
                
                radius = 4
            
                edge_index = radius_graph(pos, 
                                          radius,
                                          batch=batch.batch)
                
                x = torch.cat([x, mol_id, is_focal], dim=-1)
                
                features, coords = self.pocket_feature_extractor(h=x, 
                                                        x=pos, 
                                                        edges=edge_index, 
                                                        edge_attr=None)
            
        return features
    

    def get_policy(self, 
                   features: torch.Tensor, # (n_nodes, irreps_pocket_features)
                   batch: torch.Tensor,
                   masks: torch.Tensor = None, # (n_pockets, n_fragments)
                   ) -> Action:
            
        atom_logits = self.actor(features)
        
        if self.pocket_feature_type == 'graph':
            try:
                logits = scatter(atom_logits,
                                batch,
                                dim=0,
                                reduce='mean')
            except:
                import pdb;pdb.set_trace()
        else:
            logits = atom_logits
        
        p = 0.05
        n_m = masks.sum(dim=-1, keepdim=True)
        # n = n_m // 10 # 10% of the fragments are probable
        n = 10
        max_abs_value = torch.log((1/p - n) * (1 / (n_m - n))) / -2
        # max_abs_value = max_abs_value.broadcast_to(logits.shape)
        # logits = torch.clamp(logits, -max_abs_value, max_abs_value)
        
        # logits = self.actor(features)
        logits = logits.squeeze(-1)
        
        # Fragment sampling
        if torch.isnan(logits).any():
            print('NAN logits')
            import pdb;pdb.set_trace()
        frag_categorical = CategoricalMasked(logits=logits,
                                            masks=masks)
        
        return frag_categorical
    
    
    def get_action(self,
                   frag_categorical: CategoricalMasked,
                   frag_actions: torch.Tensor = None,
                   ) -> Action:
        
        if frag_actions is None:
            frag_actions = frag_categorical.sample()
        frag_logprob = frag_categorical.log_prob(frag_actions)
        frag_entropy = frag_categorical.entropy()
        
        action = Action(frag_i=frag_actions,
                        frag_logprob=frag_logprob,
                        frag_entropy=frag_entropy)
        
        return action
    

    def get_value(self, 
                  features: torch.Tensor,
                  batch: torch.Tensor,
                  ) -> torch.Tensor:
        
        atom_values = self.critic(features)
        
        if self.pocket_feature_type == 'graph':
            try:
                value = scatter(atom_values,
                                batch,
                                dim=0,
                                reduce='mean')
            except:
                import pdb;pdb.set_trace()
        else:
            logits = atom_values
        
        # value = scatter(atom_values,
        #                 batch,
        #                 dim=0,
        #                 reduce='mean')
        
        # value = self.critic(features)
        
        return value
    