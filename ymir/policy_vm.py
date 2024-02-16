import torch
import numpy as np
import logging

from torch.nn import Module, Sequential, Linear, ReLU
from torch.distributions import Categorical, VonMises
from ymir.model import ComENetModel
from torch import nn
from ymir.model import ComENetModel
from ymir.params import FEATURES_DIM
from torch_geometric.data import Batch

    
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
                assert masks.size() == logits.size()
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
        self.action_dim = action_dim
        self.feature_extractor = ComENetModel(features_dim=features_dim,
                                              device=device)
        self.frag_actor = nn.Sequential(nn.Linear(features_dim, features_dim),
                                         nn.ReLU(),
                                         nn.Linear(features_dim, action_dim))
        self.angle_actor = nn.Sequential(nn.Linear(features_dim, features_dim),
                                         nn.ReLU(),
                                         nn.Linear(features_dim, action_dim * 2),
                                         nn.Softplus()) 
        # angle is VonMises, so we predict mu, kappa
        
        self.critic = nn.Linear(features_dim, 1)
        self.frag_actor.to(device)
        self.angle_actor.to(device)
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
    

    def get_frag_actions(self, 
                        features: torch.Tensor, 
                        masks: torch.Tensor = None,
                        frag_actions: torch.Tensor = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.frag_actor(features)
        assert not torch.isnan(logits).any()
        categorical = CategoricalMasked(logits=logits,
                                        masks=masks)
        if frag_actions is None:
            frag_actions = categorical.sample()
        logprob = categorical.log_prob(frag_actions)
        entropy = categorical.entropy()
        return frag_actions, logprob, entropy
    
    
    def get_angle_actions(self,
                         features: torch.Tensor, 
                         angle_actions: torch.Tensor = None
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.angle_actor(features)
        n_env = params.shape[0]
        params = params.reshape(n_env, -1, 2) # should be (n_env, n_fragment, 2) where the last dim is [loc, kappa]
        locs = params[..., 0]
        kappas = params[..., 1]
        locs = torch.clamp(locs, min=-np.pi, max=np.pi)
        kappas = torch.clamp(kappas, min=0.1) # VonMises distribution has trouble sampling for low kappa values
        von_mises = VonMises(loc=locs, concentration=kappas)
        
        if angle_actions is None:
            angle_actions = von_mises.sample()
            
        logprob = von_mises.log_prob(angle_actions)
        return angle_actions, logprob


    def get_value(self, 
                  features: torch.Tensor) -> torch.Tensor:
        value = self.critic(features)
        return value