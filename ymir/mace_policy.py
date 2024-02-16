import torch
import logging

from torch.nn import Module, Sequential, Linear, ReLU
from torch.distributions import Categorical
from torch import nn
from ymir.params import FEATURES_DIM
from torch_geometric.data import Batch
from e3nn import o3
from ymir.model import MACE
from mace.modules.blocks import LinearNodeEmbeddingBlock, LinearReadoutBlock

    
class CategoricalMasked(Categorical):
    
    def __init__(self, 
                 probs: torch.Tensor = None, 
                 logits: torch.Tensor = None, 
                 validate_args: bool = None, 
                 masks: torch.Tensor = None):
        assert (probs is not None) or (logits is not None)
        self.masks = masks
        if self.masks is not None:
            if probs is not None:
                assert masks.size() == probs.size()
            elif logits is not None:
                try:
                    assert masks.size() == logits.size()
                except:
                    import pdb;pdb.set_trace()
            self.mask_value = torch.tensor(torch.finfo(logits.dtype).min, 
                                           dtype=logits.dtype)
            logits = torch.where(self.masks, logits, self.mask_value)
            
        super().__init__(probs, 
                        logits, 
                        validate_args)
    
    def entropy(self):
        if self.masks is None:
            return super().entropy()
        p_log_p = self.logits * self.probs
        zero_value = torch.tensor(0, dtype=p_log_p.dtype, 
                                           device=p_log_p.device)
        p_log_p = torch.where(self.masks, 
                              p_log_p, 
                              zero_value)
        return -p_log_p.sum(-1)
    
    
class Agent(nn.Module):
    
    def __init__(self, 
                 action_dim: int,
                 hidden_irreps: o3.Irreps,
                 num_elements: int,
                 device: torch.device = torch.device('cuda')
                 ):
        super(Agent, self).__init__()
        self.feature_extractor = MACE(hidden_irreps=hidden_irreps,
                                      num_elements=num_elements)
        self.feature_extractor = self.feature_extractor.to(device)
        # self.irreps_out_actor = o3.Irreps([(action_dim, (0, 1))])
        # self.irreps_out_critic = o3.Irreps([(1, (0, 1))])
        
        # import pdb;pdb.set_trace()
        
        # self.irreps_scalars_dim = hidden_irreps[0].dim
        # self.irreps_vectors_dim = hidden_irreps[1].dim
        
        # self.actor = Linear(in_features=self.irreps_vectors_dim,
        #                out_features=action_dim)
        # self.critic = Linear(in_features=self.irreps_scalars_dim,
        #                 out_features=1)
        
        self.actor = Linear(in_features=hidden_irreps.dim,
                       out_features=action_dim)
        self.critic = Linear(in_features=hidden_irreps.dim,
                        out_features=1)
        
        # self.actor = LinearNodeEmbeddingBlock(irreps_in=hidden_irreps,
        #                                     irreps_out=self.irreps_out_actor)
        # # import pdb;pdb.set_trace()
        # self.critic = LinearNodeEmbeddingBlock(irreps_in=hidden_irreps,
        #                                     irreps_out=self.irreps_out_critic)
        self.actor.to(device)
        self.critic.to(device)

    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        actions, logprobs, entropies = self.get_action(features, 
                                  masks)
        value = self.get_value(features)
        return actions, value


    def extract_features(self,
                         x: Batch):
        features = self.feature_extractor(x)
        return features
    

    def get_action(self, 
                   features: torch.Tensor, 
                   masks: torch.Tensor = None,
                   actions: torch.Tensor = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inv_features, equi_features = features.split([self.irreps_scalars_dim, self.irreps_vectors_dim], dim=-1)
        # logits = self.actor(equi_features)
        logits = self.actor(features)
        # logging.info(logits)
        if torch.isnan(logits).any():
            print('NAN logits')
            import pdb;pdb.set_trace()
        categorical = CategoricalMasked(logits=logits,
                                        masks=masks)
        if actions is None:
            actions = categorical.sample()
        logprob = categorical.log_prob(actions)
        entropy = categorical.entropy()
        return actions, logprob, entropy


    def get_value(self, 
                  features: torch.Tensor) -> torch.Tensor:
        # inv_features, equi_features = features.split([self.irreps_scalars_dim, self.irreps_vectors_dim], dim=-1)
        # value = self.critic(inv_features)
        value = self.critic(features)
        return value