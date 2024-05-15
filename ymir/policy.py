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
                         TORSION_ANGLES_DEG)
from ymir.atomic_num_table import AtomicNumberTable
from ymir.data.fragment import Fragment
from torch_geometric.data import Batch
from ymir.model import CNN
from torch_scatter import scatter
from ymir.featurizer_sn import Featurizer
from torch_geometric.data import Data
    
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
                #  features_dim: int,
                 hidden_irreps: o3.Irreps = HIDDEN_IRREPS,
                 irreps_output: o3.Irreps = IRREPS_OUTPUT,
                #  lmax: int = L_MAX,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 torsion_angle_deg: list[float] = TORSION_ANGLES_DEG,
                 ):
        super(Agent, self).__init__()
        self.protected_fragments = protected_fragments
        self.z_table = atomic_num_table
        # self.features_dim = features_dim
        self.hidden_irreps = hidden_irreps
        self.irreps_output = irreps_output
        # self.lmax = lmax
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        self.torsion_angles_deg = torsion_angle_deg
        
        self.n_fragments = len(self.protected_fragments)
        self.action_dim = self.n_fragments
        
        data_list = []
        featurizer = Featurizer(z_table=self.z_table)
        for fragment in self.protected_fragments:
            x, pos, is_focal = featurizer.get_fragment_features(fragment=fragment, 
                                                                embed_hydrogens=self.embed_hydrogens,
                                                                center_pos=[0, 0, 0])
            
            x = torch.tensor(x, dtype=torch.long)
            pos = torch.tensor(pos, dtype=torch.float)
            is_focal = torch.tensor(is_focal, dtype=torch.bool)
            mol_id = torch.tensor([2] * len(x), dtype=torch.long)
            data = Data(x=x, 
                        pos=pos,
                        mol_id=mol_id,
                        is_focal=is_focal)
            data_list.append(data)
        self.fragment_batch = Batch.from_data_list(data_list)
        self.fragment_batch = self.fragment_batch.to(self.device)
        
        self.pocket_irreps = o3.Irreps(f'1x0e + 8x1o + 8x2e')
        self.pocket_feature_extractor = CNN(hidden_irreps=o3.Irreps(f'16x1o + 16x2e'),
                                            irreps_output=self.pocket_irreps,
                                             num_elements=len(self.z_table))
        
        self.fragment_irreps = o3.Irreps(f'4x1o + 4x2e')
        self.fragment_feature_extractor = CNN(hidden_irreps=o3.Irreps(f'8x1o + 8x2e'),
                                              irreps_output=self.fragment_irreps,
                                              num_elements=len(self.z_table))
        
        actor_irreps = o3.Irreps(f'1x0e')
        self.actor_tp = o3.FullyConnectedTensorProduct(irreps_in1=self.pocket_irreps[1:], 
                                                            irreps_in2=self.fragment_irreps, 
                                                            irreps_out=actor_irreps, 
                                                            internal_weights=True)
        self.actor_tp.to(self.device)
        
        self.torsion_angles_rad = torch.deg2rad(torch.tensor(self.torsion_angles_deg))
        self.rot_mats = o3.matrix_x(self.torsion_angles_rad)
        self.d_hidden_irreps = self.fragment_irreps.D_from_matrix(self.rot_mats) # (n_angles, irreps_dim, irreps_dim)
        self.d_hidden_irreps = self.d_hidden_irreps.float().to(self.device)


    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        action = self.get_action(features, masks)
        value = self.get_value(features)
        return action, value
    
    
    def extract_features(self,
                         x: Batch):
        # noise = torch.randn_like(x.pos) / 50 # variance of 0.1
        # noise = noise.to(x.pos)
        # x.pos = x.pos + noise
        features = self.pocket_feature_extractor(x)
        return features
    
    
    def extract_fragment_features(self):
        features = self.fragment_feature_extractor(self.fragment_batch)
        
        fragment_features = torch.einsum('bi,rji->brj', features, self.d_hidden_irreps)
        fragment_features = fragment_features.reshape(-1, self.fragment_irreps.dim)
        
        # apply rotation
        # n_frags = len(features)
        # n_rotations = len(self.torsion_angles_rad)
        # frag_indices = torch.arange(n_frags)
        # frag_indices = torch.repeat_interleave(frag_indices, n_rotations)
        # frag_features = features[frag_indices] # (n_pocket*n_rotations*n_fragments, irreps_dim)
        
        # rotation_indices = torch.arange(n_rotations)
        # rotation_indices = rotation_indices.repeat(n_frags)
        # rotations = self.d_hidden_irreps[rotation_indices]
        
        return fragment_features
    

    def get_policy(self, 
                   features: torch.Tensor, # (n_nodes, irreps_pocket_features)
                   fragment_features: torch.Tensor,
                   masks: torch.Tensor = None, # (n_pockets, n_fragments)
                #    frag_actions: torch.Tensor = None # (action_dim),
                   ) -> Action:
            
        inv_features, equi_features = self.split_features(features, self.pocket_irreps)
        
        n_pockets = equi_features.shape[0]
        n_fragments = fragment_features.shape[0]
        # [pocket1, pocket1, ..., pocket2..., pocketi] * n_fragment
        pocket_indices = torch.arange(n_pockets)
        pocket_indices = torch.repeat_interleave(pocket_indices, n_fragments)
        pocket_embeddings = equi_features[pocket_indices] # (n_pocket*n_rotations*n_fragments, irreps_dim)
        
        # [fragment1, fragment2, fragment3,..., fragmentn-1, fragmentn]
        fragment_indices = torch.arange(n_fragments)
        fragment_indices = fragment_indices.repeat(n_pockets)
        fragment_embeddings = fragment_features[fragment_indices]
        
        actor_output = self.actor_tp(pocket_embeddings, fragment_embeddings)
        
        frag_logits = actor_output.reshape(n_pockets, n_fragments)
        
        # frag_logits = torch.tanh(frag_logits)
        # frag_logits = frag_logits * 4 # from [-1;1] to [-3;3]
        # if 2000 fragments, the maximum prob will be exp(3) / (exp(-3) *700 + exp(3)) = 0.30
        # frag_logits = frag_logits - frag_logits.max(axis=1).values.reshape(-1, 1)
        # frag_logits = torch.clamp(frag_logits, -2, 2)
        
        # Fragment sampling
        if torch.isnan(frag_logits).any():
            print('NAN logits')
            import pdb;pdb.set_trace()
        frag_categorical = CategoricalMasked(logits=frag_logits,
                                            masks=masks)
        
        return frag_categorical
        
        if frag_actions is None:
            frag_actions = frag_categorical.sample()
        frag_logprob = frag_categorical.log_prob(frag_actions)
        frag_entropy = frag_categorical.entropy()
        
        action = Action(frag_i=frag_actions,
                        frag_logprob=frag_logprob,
                        frag_entropy=frag_entropy)
        
        return action
    
    
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
                  ) -> torch.Tensor:
        inv_features, equi_features = self.split_features(features, self.pocket_irreps)
        # critic_output = self.critic(inv_features)
        # critic_output = self.critic(features)
        value = inv_features.squeeze(-1)
        return value
    
    
    def split_features(self,
                       features: torch.Tensor,
                       irreps: o3.Irreps
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        slices = irreps.slices()
        inv_slice = slices[0]
        inv_features = features[..., inv_slice]
        equi_slice = slice(slices[1].start, slices[-1].stop)
        equi_features = features[..., equi_slice]
        return inv_features, equi_features