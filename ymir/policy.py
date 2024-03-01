import torch

from torch import nn
from torch_geometric.data import Batch
from e3nn import o3
from ymir.distribution import CategoricalMasked
from typing import NamedTuple
from ymir.params import (EMBED_HYDROGENS, 
                         LMAX, 
                         HIDDEN_IRREPS,
                         IRREPS_OUTPUT)
from ymir.atomic_num_table import AtomicNumberTable
from ymir.data.fragment import Fragment
from torch_geometric.data import Batch
from ymir.model import CNN
    
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
            modules.append(nn.Dropout(p=0.25))
        modules.append(nn.Linear(dims[-1], output_dim))
        # modules.append(nn.Tanh())

        self.sequential = nn.Sequential(*modules)
        
    def forward(self,
                x: torch.Tensor):
        return self.sequential(x)
    
    
class Agent(nn.Module):
    
    def __init__(self, 
                 protected_fragments: list[Fragment],
                 atomic_num_table: AtomicNumberTable,
                #  features_dim: int = FEATURES_DIM,
                 hidden_irreps: o3.Irreps = HIDDEN_IRREPS,
                 irreps_output: o3.Irreps = IRREPS_OUTPUT,
                 lmax: int = LMAX,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 ):
        super(Agent, self).__init__()
        self.protected_fragments = protected_fragments
        self.z_table = atomic_num_table
        self.hidden_irreps = hidden_irreps
        self.irreps_output = irreps_output
        self.lmax = lmax
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        
        self.n_fragments = len(self.protected_fragments)
        self.action_dim = self.n_fragments
        
        # self.pocket_feature_extractor = ComENetModel(config=COMENET_CONFIG,
        #                                              features_dim=self.features_dim,
        #                                              readout='mean')
        # # Fragment logit + 3D vector 
        # hidden_dims = [self.features_dim, 
        #                self.features_dim // 2]
        # self.actor = MultiLinear(input_dim=self.features_dim, 
        #                          hidden_dims=hidden_dims,
        #                          output_dim=self.action_dim)
        
        # self.critic = MultiLinear(input_dim=self.features_dim, 
        #                          hidden_dims=hidden_dims,
        #                          output_dim=1)
        
        self.pocket_feature_extractor = CNN(hidden_irreps=self.hidden_irreps,
                                            irreps_output=self.irreps_output,
                                             num_elements=len(self.z_table))
        
        assert(len(self.irreps_output) == 1) # only invariant features
        self.inv_features_dim = self.irreps_output.dim
        hidden_dims = [self.inv_features_dim * 2, 
                    #    self.inv_features_dim, 
                    #    self.inv_features_dim // 2
                       ]
        self.actor = MultiLinear(input_dim=self.inv_features_dim, 
                                 hidden_dims=hidden_dims,
                                 output_dim=self.action_dim)
        
        self.critic = MultiLinear(input_dim=self.inv_features_dim, 
                                 hidden_dims=hidden_dims,
                                 output_dim=1)


    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        action = self.get_action(features, masks)
        value = self.get_value(features)
        return action, value


    def extract_features(self,
                         x: Batch):
        noise = torch.randn_like(x.pos) / 10 # variance of 0.1
        noise = noise.to(x.pos)
        x.pos = x.pos + noise
        features = self.pocket_feature_extractor(x)
        return features
    

    def get_action(self, 
                   features: torch.Tensor, # (n_pockets, irreps_pocket_features)
                   masks: torch.Tensor = None, # (n_pockets, n_fragments)
                   frag_actions: torch.Tensor = None # (action_dim)
                   ) -> Action:
            
        # inv_features, equi_features = self.split_features(features, self.hidden_irreps)
        frag_logits = self.actor(features)
        frag_logits = torch.tanh(frag_logits)
        frag_logits = frag_logits * 3 # from [-1;1] to [-3;3]
        # if 2000 fragments, the maximum prob will be exp(3) / (exp(-3) *700 + exp(3)) = 0.30
        
        # Fragment sampling
        if torch.isnan(frag_logits).any():
            print('NAN logits')
            import pdb;pdb.set_trace()
        frag_categorical = CategoricalMasked(logits=frag_logits,
                                            masks=masks)
        if frag_actions is None:
            frag_actions = frag_categorical.sample()
        frag_logprob = frag_categorical.log_prob(frag_actions)
        frag_entropy = frag_categorical.entropy()
        
        action = Action(frag_i=frag_actions,
                        frag_logprob=frag_logprob,
                        frag_entropy=frag_entropy)
        
        return action


    def get_value(self, 
                  features: torch.Tensor) -> torch.Tensor:
        # inv_features, equi_features = self.split_features(features, self.hidden_irreps)
        # critic_output = self.critic(inv_features)
        critic_output = self.critic(features)
        value = critic_output.squeeze(-1)
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