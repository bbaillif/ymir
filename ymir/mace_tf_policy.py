import torch
import logging

from torch import nn
from torch_geometric.data import Batch
from e3nn import o3
from ymir.model import Transformer
from ymir.distribution import CategoricalMasked
from typing import NamedTuple
from pyro.distributions import ProjectedNormal
# from scipy.stats import vonmises_fisher
    
class Action(NamedTuple): 
    frag_i: torch.Tensor # (batch_size)
    frag_logprob: torch.Tensor # (batch_size)
    frag_entropy: torch.Tensor # (batch_size)
    vector: torch.Tensor # (batch_size, 3)
    vector_logprob: torch.Tensor # (batch_size)
    
class Agent(nn.Module):
    
    def __init__(self, 
                 action_dim: int,
                 hidden_irreps: o3.Irreps,
                 device: torch.device = torch.device('cuda')
                 ):
        super(Agent, self).__init__()
        self.action_dim = action_dim
        self.hidden_irreps = hidden_irreps
        self.device = device
        
        self.feature_extractor = Transformer(irreps_output=self.hidden_irreps)
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # Fragment logit + 3D vector 
        self.actor_irreps = o3.Irreps(f'{self.action_dim}x0e + {self.action_dim}x1o')
        self.actor = o3.Linear(self.hidden_irreps, self.actor_irreps)
        
        self.critic_irreps = o3.Irreps(f'1x0e')
        self.critic = o3.Linear(self.hidden_irreps, self.critic_irreps)
        
        self.actor.to(self.device)
        self.critic.to(self.device)

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
                   frag_actions: torch.Tensor = None,
                   vector_actions: torch.Tensor = None,
                   ) -> Action:
        
        actor_output = self.actor(features)
        inv_features, equi_features = actor_output.split([self.action_dim, self.action_dim * 3], dim=-1)
        
        # Fragment sampling
        frag_logits = inv_features
        if torch.isnan(frag_logits).any():
            print('NAN logits')
            import pdb;pdb.set_trace()
        frag_categorical = CategoricalMasked(logits=frag_logits,
                                            masks=masks)
        if frag_actions is None:
            frag_actions = frag_categorical.sample()
        frag_logprob = frag_categorical.log_prob(frag_actions)
        frag_entropy = frag_categorical.entropy()
        
        # Direction selection
        # concentrations = equi_features.norm(dim=-1)
        # normalized_vectors = equi_features / concentrations
        # vmf_distribution = vonmises_fisher(mu=normalized_vectors,
        #                                    kappa=concentrations)
        # if vector_actions is None:
        #     vector_actions = vmf_distribution()
        
        equi_features = equi_features.reshape(equi_features.shape[0], -1, 3)
        
        # Direction sampling
        # We consider equivariant features are vectors with direction and magnitude equal to concentration
        proj_norm = ProjectedNormal(concentration=equi_features)
        if vector_actions is None:
            vector_actions = proj_norm.sample() 
            # we could use rsample to differentiate on this reparametrization, 
            # but we would need a differentiable reward
        vector_logprob = proj_norm.log_prob(vector_actions)
        
        action = Action(frag_i=frag_actions,
                        frag_logprob=frag_logprob,
                        frag_entropy=frag_entropy,
                        vector=vector_actions,
                        vector_logprob=vector_logprob)
        
        return action


    def get_value(self, 
                  features: torch.Tensor) -> torch.Tensor:
        value = self.critic(features)
        return value