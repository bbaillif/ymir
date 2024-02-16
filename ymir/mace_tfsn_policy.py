import torch
import logging

from torch import nn
from torch_geometric.data import (Batch, 
                                  Data)
from e3nn import o3
from ymir.model import TransformerSN
from ymir.distribution import CategoricalMasked
from typing import NamedTuple
from pyro.distributions import ProjectedNormal
from ymir.data.fragment import Fragment
from ymir.featurizer_sn import get_mol_features
# from scipy.stats import vonmises_fisher
    
class Action(NamedTuple): 
    frag_i: torch.Tensor # (batch_size)
    frag_logprob: torch.Tensor # (batch_size)
    frag_entropy: torch.Tensor # (batch_size)
    vector: torch.Tensor # (batch_size, 3)
    vector_logprob: torch.Tensor # (batch_size)
    
class Agent(nn.Module):
    
    def __init__(self, 
                 protected_fragments: list[Fragment],
                 hidden_irreps: o3.Irreps,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = False,
                 ):
        super(Agent, self).__init__()
        self.protected_fragments = protected_fragments
        self.hidden_irreps = hidden_irreps
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        
        self.action_dim = len(self.protected_fragments)
        center_pos = [0, 0, 0]
        
        data_list = []
        for fragment in self.protected_fragments:
            x, pos = get_mol_features(mol=fragment, 
                                        embed_hydrogens=self.embed_hydrogens,
                                        center_pos=center_pos)
            x = torch.tensor(x)
            pos = torch.tensor(pos)
            mol_id = torch.tensor([2] * len(x))
            data = Data(x=x, 
                        pos=pos,
                        mol_id=mol_id)
            data_list.append(data)
        self.fragment_batch = Batch.from_data_list(data_list)
        self.fragment_batch = self.fragment_batch.to(self.device)
        
        self.feature_extractor = TransformerSN(irreps_output=self.hidden_irreps)
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # Fragment logit + 3D vector 
        self.actor_irreps = o3.Irreps(f'1x0e + 1x1o')
        self.actor_tp = o3.FullyConnectedTensorProduct(irreps_in1=hidden_irreps, 
                                                        irreps_in2=hidden_irreps, 
                                                        irreps_out=self.actor_irreps, 
                                                        internal_weights=True)
        
        self.critic_irreps = o3.Irreps(f'1x0e')
        self.critic = o3.Linear(self.hidden_irreps, self.critic_irreps)
        
        self.actor_tp.to(self.device)
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
    
    
    def extract_fragment_features(self):
        fragment_features = self.feature_extractor(self.fragment_batch)
        return fragment_features

    def get_action(self, 
                   features: torch.Tensor, 
                   fragment_features: torch.Tensor,
                   masks: torch.Tensor = None,
                   frag_actions: torch.Tensor = None,
                   vector_actions: torch.Tensor = None,
                   ) -> Action:
        
        n_pockets = features.shape[0]
        n_fragments = fragment_features.shape[0]
        
        # Repeat the vectors such as shape[0] is (n_pockets, n_fragments)
        # Since the Tensorproduct does not work with 3 dimensions
        # [pocket1, pocket2, pocket3] * n_fragment
        pocket_indices = torch.arange(n_pockets)
        pocket_indices = torch.repeat_interleave(pocket_indices, n_fragments)
        pocket_embeddings = features[pocket_indices]
        
        # [fragment1, fragment1, fragment1,..., fragmentn, fragmentn, fragmentn]
        fragment_indices = torch.arange(n_fragments)
        fragment_indices = fragment_indices.repeat(n_pockets)
        fragment_embeddings = fragment_features[fragment_indices]
        
        actor_output = self.actor_tp(pocket_embeddings, fragment_embeddings)
        actor_output = actor_output.reshape(n_pockets, n_fragments, 4) # -1 should be 4
        
        inv_features, equi_features = actor_output.split([1, 3], dim=-1)
        
        # Fragment sampling
        frag_logits = inv_features.squeeze(-1)
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