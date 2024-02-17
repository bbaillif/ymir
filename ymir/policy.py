import torch
import logging
import numpy as np

from torch import nn
from torch_geometric.data import (Batch, 
                                  Data)
from e3nn import o3
from ymir.model import TransformerSN
from ymir.distribution import CategoricalMasked
from typing import NamedTuple, Sequence
from ymir.data.fragment import Fragment
from ymir.featurizer_sn import get_fragment_features
from ymir.params import EMBED_HYDROGENS, HIDDEN_IRREPS, TORSION_ANGLES_DEG
# from scipy.stats import vonmises_fisher
    
class Action(NamedTuple): 
    frag_i: torch.Tensor # (batch_size)
    frag_logprob: torch.Tensor # (batch_size)
    frag_entropy: torch.Tensor # (batch_size)
    
class Agent(nn.Module):
    
    def __init__(self, 
                 protected_fragments: list[Fragment],
                 hidden_irreps: o3.Irreps = HIDDEN_IRREPS,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 summing_embeddings: bool = False,
                 torsion_angles_deg: Sequence[float] = TORSION_ANGLES_DEG,
                 ):
        super(Agent, self).__init__()
        self.protected_fragments = protected_fragments
        self.hidden_irreps = hidden_irreps
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        self.summing_embeddings = summing_embeddings
        self.torsion_angles_deg = torsion_angles_deg
        self.n_rotations = len(self.torsion_angles_deg)
        
        self.action_dim = len(self.protected_fragments)
        center_pos = [0, 0, 0]
        
        data_list = []
        for fragment in self.protected_fragments:
            x, pos = get_fragment_features(fragment=fragment, 
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
        self.actor_irreps = o3.Irreps(f'1x0e')
        if self.summing_embeddings:
            self.actor = o3.Linear(self.hidden_irreps, self.actor_irreps)
            self.actor.to(self.device)
        else:
            # hidden_irreps[1:] is the equivariant part only
            self.actor_tp = o3.FullyConnectedTensorProduct(irreps_in1=hidden_irreps[1:], 
                                                            irreps_in2=hidden_irreps[1:], 
                                                            irreps_out=self.actor_irreps, 
                                                            internal_weights=True)
            self.actor_tp.to(self.device)
        
        self.critic_irreps = o3.Irreps(f'1x0e')
        self.critic = o3.Linear(self.hidden_irreps[:1], self.critic_irreps)
        self.critic.to(self.device)
        
        # Here we compute the negative torsion angle because the torsion angle refers 
        # to fragment rotation, and here we rotate the pockets
        self.neg_torsion_angles_deg = -torch.tensor(self.torsion_angles_deg)
        self.neg_torsion_angles_rad = torch.deg2rad(self.neg_torsion_angles_deg)
        self.rot_mats = o3.matrix_x(self.neg_torsion_angles_rad)
        self.d_hidden_irreps = self.hidden_irreps.D_from_matrix(self.rot_mats) # (n_angles, irreps_dim, irreps_dim)
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
        
        # Apply pocket rotations
        try:
            pocket_embeddings = torch.einsum('bi,rji->brj', features, self.d_hidden_irreps)
        except:
            import pdb;pdb.set_trace()
        pocket_embeddings = pocket_embeddings.reshape(-1, self.hidden_irreps.dim) # (n_pocket*n_rotations, irreps_dim)
        
        n_pocket_rot = n_pockets * self.n_rotations
        
        # [pocket1rot1, pocket1rot2, ..., pocket2rot1..., pocketirotk] * n_fragment
        pocket_indices = torch.arange(n_pocket_rot)
        pocket_indices = torch.repeat_interleave(pocket_indices, n_fragments)
        pocket_embeddings = pocket_embeddings[pocket_indices] # (n_pocket*n_rotations*n_fragments, irreps_dim)
        
        # [fragment1, fragment1, fragment1,..., fragmentn, fragmentn, fragmentn]
        fragment_indices = torch.arange(n_fragments)
        fragment_indices = fragment_indices.repeat(n_pocket_rot)
        fragment_embeddings = fragment_features[fragment_indices]
        
        # Embedding dimension is
        # Pocket0Rot0Frag0, Pocket0Rot0Frag1, ... Pocket0Rot1Frag0, Pocket0Rot1Frag1...
        
        _, pocket_equi_embeddings = self.split_features(pocket_embeddings)
        _, fragment_equi_embeddings = self.split_features(fragment_embeddings)
        
        if self.summing_embeddings:
            actor_output = self.actor(pocket_embeddings + fragment_embeddings)
        else:
            actor_output = self.actor_tp(pocket_equi_embeddings, fragment_equi_embeddings)
            
        actor_output = actor_output.reshape(n_pockets, self.n_rotations, n_fragments)
        actor_output = actor_output.transpose(-2, -1) # Transpose "rotation" and "fragment" in the matrix
        actor_output = actor_output.reshape(n_pockets, -1) #last dim is Frag0Rot0 Frag0Rot1...
        
        masks = torch.repeat_interleave(masks, self.n_rotations, dim=1)
        
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
        inv_features, _ = self.split_features(features)
        value = self.critic(inv_features)
        return value
    
    
    def split_features(self,
                       features: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        slices = self.hidden_irreps.slices()
        inv_slice = slices[0]
        inv_features = features[..., inv_slice]
        equi_slice = slice(slices[1].start, slices[-1].stop)
        equi_features = features[..., equi_slice]
        return inv_features, equi_features