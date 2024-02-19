import torch

from torch import nn
from torch_geometric.data import Batch
from e3nn import o3
from ymir.y_feature_extractor import YFeatureExtractor
from ymir.distribution import CategoricalMasked
from typing import NamedTuple
from ymir.params import (EMBED_HYDROGENS, 
                         LMAX, 
                         MAX_RADIUS)
from ymir.atomic_num_table import AtomicNumberTable
from pyro.distributions import ProjectedNormal
    
class Action(NamedTuple): 
    frag_i: torch.Tensor # (batch_size)
    frag_logprob: torch.Tensor # (batch_size)
    frag_entropy: torch.Tensor # (batch_size)
    vector: torch.Tensor # (batch_size, 3)
    vector_logprob: torch.Tensor # (batch_size)
    
class Agent(nn.Module):
    
    def __init__(self, 
                 action_dim: int,
                 atomic_num_table: AtomicNumberTable,
                 lmax: int = LMAX,
                 max_radius: float = MAX_RADIUS,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 summing_embeddings: bool = False
                 ):
        super(Agent, self).__init__()
        self.action_dim = action_dim
        self.z_table = atomic_num_table
        self.lmax = lmax
        self.max_radius = max_radius
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        self.summing_embeddings = summing_embeddings
        
        self.feature_extractor = YFeatureExtractor(atomic_num_table=self.z_table,
                                                   lmax=self.lmax,
                                                   max_radius=self.max_radius,
                                                   summing_embeddings=self.summing_embeddings)
        self.irreps_features = self.feature_extractor.irreps_out
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # Fragment logit + 3D vector 
        self.actor_irreps = o3.Irreps(f'{self.action_dim}x0e + {self.action_dim}x1o')
        self.actor = o3.Linear(irreps_in=self.irreps_features, 
                                irreps_out=self.actor_irreps)
        self.actor.to(self.device)
        
        self.critic_irreps = o3.Irreps(f'1x0e')
        self.critic = o3.Linear(irreps_in=self.irreps_features, 
                                irreps_out=self.critic_irreps)
        self.critic.to(self.device)


    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        action = self.get_action(features, masks)
        value = self.get_value(features)
        return action, value


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
        
        slices = self.actor_irreps.slices()
        inv_slice = slices[0]
        inv_features = actor_output[..., inv_slice]
        equi_slice = slice(slices[1].start, slices[-1].stop)
        equi_features = actor_output[..., equi_slice]
        
        vectors = equi_features.reshape(equi_features.shape[0], -1, 3)
        # norms = vectors.norm(dim=-1)
        
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
        
        # Direction sampling
        # We consider equivariant features are vectors with direction and magnitude equal to concentration
        proj_norm = ProjectedNormal(concentration=vectors)
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
        critic_output = self.critic(features)
        value = critic_output
        return value
    