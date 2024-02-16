import torch
import logging

from torch.nn import Module, Sequential, Linear, ReLU
from torch.distributions import Categorical
from ymir.model import ComENetModel
from torch import nn
from ymir.model import ComENetModel
from ymir.params import FEATURES_DIM
from torch_geometric.data import Batch

class Actor(Module):
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, 
                         **kwargs)
        
        self.lin1 = Linear(in_features=input_dim,
                           out_features=output_dim)
                            # out_features=hidden_dim)
                            
        # self.act = ReLU()
        # self.lin2 = Linear(in_features=hidden_dim,
        #                     out_features=output_dim)
        # self.nns = Sequential(self.lin1,
        #                         self.act,
        #                         self.lin2)
        
    def forward(self,
                x):
        # return self.nns(x)
        output = self.lin1(x)
        return output
    
    
class Critic(Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, 
                         **kwargs)
        
        self.lin1 = Linear(in_features=input_dim,
                            out_features=hidden_dim)
        self.act = ReLU()
        self.lin2 = Linear(in_features=hidden_dim,
                            out_features=output_dim)
        self.nns = Sequential(self.lin1,
                                self.act,
                                self.lin2)
        
    def forward(self,
                x):
        return self.nns(x)
    
    
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
                 features_dim: int = FEATURES_DIM,
                 device: torch.device = torch.device('cuda')
                 ):
        super(Agent, self).__init__()
        self.feature_extractor = ComENetModel(features_dim=features_dim,
                                              device=device)
        self.actor = Actor(input_dim=features_dim,
                           output_dim=action_dim)
        self.critic = Critic(input_dim=features_dim)
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
        logits = self.actor(features)
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
        value = self.critic(features)
        return value