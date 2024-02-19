import torch
import logging
import numpy as np

from torch import nn
from torch_geometric.data import (Batch, 
                                  Data)
from e3nn import o3
from ymir.y_feature_extractor import YFeatureExtractor
from ymir.distribution import CategoricalMasked
from typing import NamedTuple
from ymir.data.fragment import Fragment
from ymir.featurizer_sn import Featurizer
from ymir.params import (EMBED_HYDROGENS, 
                         HIDDEN_IRREPS, 
                         LMAX, 
                         MAX_RADIUS)
from ymir.atomic_num_table import AtomicNumberTable
# from scipy.stats import vonmises_fisher
    
class Action(NamedTuple): 
    frag_i: torch.Tensor # (batch_size)
    frag_logprob: torch.Tensor # (batch_size)
    frag_entropy: torch.Tensor # (batch_size)
    
class Agent(nn.Module):
    
    def __init__(self, 
                 protected_fragments: list[Fragment],
                 atomic_num_table: AtomicNumberTable,
                 lmax: int = LMAX,
                 max_radius: float = MAX_RADIUS,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 summing_embeddings: bool = False
                 ):
        super(Agent, self).__init__()
        self.protected_fragments = protected_fragments
        self.z_table = atomic_num_table
        self.lmax = lmax
        self.max_radius = max_radius
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        self.summing_embeddings = summing_embeddings
        
        self.action_dim = len(self.protected_fragments)
        self.featurizer = Featurizer(z_table=self.z_table)
        
        center_pos = [0, 0, 0]
        data_list = []
        for fragment in self.protected_fragments:
            x, pos = self.featurizer.get_fragment_features(fragment=fragment, 
                                                            embed_hydrogens=self.embed_hydrogens,
                                                            center_pos=center_pos)
            x = torch.tensor(x)
            pos = torch.tensor(pos)
            data = Data(x=x, 
                        pos=pos)
            data_list.append(data)
            
        self.fragment_batch = Batch.from_data_list(data_list)
        self.fragment_batch = self.fragment_batch.to(self.device)
        
        self.feature_extractor = YFeatureExtractor(atomic_num_table=self.z_table,
                                                   lmax=self.lmax,
                                                   max_radius=self.max_radius,
                                                   summing_embeddings=self.summing_embeddings)
        self.irreps_features = self.feature_extractor.irreps_out
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # Fragment logit + 3D vector 
        self.actor_irreps = o3.Irreps(f'1x0e')
        if self.summing_embeddings:
            self.actor_ts = o3.TensorSquare(self.irreps_features)
            self.actor_ts.to(self.device)
            self.actor_linear = o3.Linear(irreps_in=self.actor_ts.irreps_out, 
                                            irreps_out=self.actor_irreps,
                                            biases=True)
            self.actor_linear.to(self.device)
        else:
            # hidden_irreps[1:] is the equivariant part only
            self.actor_tp = o3.FullyConnectedTensorProduct(irreps_in1=self.irreps_features[1:], 
                                                            irreps_in2=self.irreps_features[1:], 
                                                            irreps_out=self.actor_irreps, 
                                                            internal_weights=True)
            self.actor_tp.to(self.device)
        
        self.critic_irreps = o3.Irreps(f'1x0e')
        self.critic_ts = o3.TensorSquare(self.irreps_features)
        self.critic_linear = o3.Linear(irreps_in=self.critic_ts.irreps_out, 
                                       irreps_out=self.critic_irreps,
                                       biases=True)
        self.critic_ts.to(self.device)
        self.critic_linear.to(self.device)


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
                   ) -> Action:
        
        n_pockets = features.shape[0]
        n_fragments = fragment_features.shape[0]
        
        # [pocket1, pocket2, pocketi] * n_fragment
        pocket_indices = torch.arange(n_pockets)
        pocket_indices = torch.repeat_interleave(pocket_indices, n_fragments)
        pocket_embeddings = features[pocket_indices] # (n_pocket*n_rotations*n_fragments, irreps_dim)
        
        # [fragment1, fragment1, fragment1,..., fragmentn, fragmentn, fragmentn]
        fragment_indices = torch.arange(n_fragments)
        fragment_indices = fragment_indices.repeat(n_pockets)
        fragment_embeddings = fragment_features[fragment_indices]
        
        # Embedding dimension is
        # Pocket0Frag0, Pocket0Frag1, ... Pocket1Frag0, Pocket1Frag1...
        
        _, pocket_equi_embeddings = self.split_features(pocket_embeddings)
        _, fragment_equi_embeddings = self.split_features(fragment_embeddings)
        
        if self.summing_embeddings:
            actor_output = self.actor_ts(pocket_embeddings + fragment_embeddings)
            actor_output = self.actor_linear(actor_output)
        else:
            actor_output = self.actor_tp(pocket_equi_embeddings, fragment_equi_embeddings)
            
        actor_output = actor_output.reshape(n_pockets, n_fragments)
        
        # Fragment sampling
        frag_logits = actor_output
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
        critic_output = self.critic_ts(features)
        critic_output = self.critic_linear(critic_output)
        value = critic_output
        return value
    
    
    def split_features(self,
                       features: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        slices = self.irreps_features.slices()
        inv_slice = slices[0]
        inv_features = features[..., inv_slice]
        equi_slice = slice(slices[1].start, slices[-1].stop)
        equi_features = features[..., equi_slice]
        return inv_features, equi_features