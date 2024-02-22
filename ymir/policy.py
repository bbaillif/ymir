import torch

from torch import nn
from torch_geometric.data import Batch
from e3nn import o3
from ymir.model.transformer import Transformer
from ymir.distribution import CategoricalMasked
from typing import NamedTuple, Sequence
from ymir.params import (EMBED_HYDROGENS, 
                         LMAX, 
                         MAX_RADIUS,
                         TORSION_ANGLES_DEG,
                         HIDDEN_IRREPS,
                         IRREPS_FRAGMENTS)
from ymir.atomic_num_table import AtomicNumberTable
from ymir.data.fragment import Fragment
from pyro.distributions import ProjectedNormal
    
class Action(NamedTuple): 
    frag_i: torch.Tensor # (batch_size)
    frag_logprob: torch.Tensor # (batch_size)
    frag_entropy: torch.Tensor # (batch_size)
    
class Agent(nn.Module):
    
    def __init__(self, 
                 protected_fragments: list[Fragment],
                 atomic_num_table: AtomicNumberTable,
                 irreps_pocket_features: o3.Irreps = HIDDEN_IRREPS,
                 irreps_fragment_features: o3.Irreps = IRREPS_FRAGMENTS,
                 lmax: int = LMAX,
                 max_radius: float = MAX_RADIUS,
                 device: torch.device = torch.device('cuda'),
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 torsion_angles_deg: Sequence[float] = TORSION_ANGLES_DEG,
                 ):
        super(Agent, self).__init__()
        self.protected_fragments = protected_fragments
        self.z_table = atomic_num_table
        self.irreps_pocket_features = irreps_pocket_features
        self.irreps_fragment_features = irreps_fragment_features
        self.lmax = lmax
        self.max_radius = max_radius
        self.device = device
        self.embed_hydrogens = embed_hydrogens
        self.torsion_angles_deg = torsion_angles_deg
        
        self.n_rotations = len(self.torsion_angles_deg)
        self.n_fragments = len(self.protected_fragments)
        self.action_dim = self.n_fragments * self.n_rotations
        
        self.pocket_feature_extractor = Transformer(irreps_output=self.irreps_pocket_features,
                                                    num_elements=len(self.z_table),
                                                    max_radius=self.max_radius)
        self.pocket_feature_extractor = self.pocket_feature_extractor.to(self.device)
        
        # self.fragment_feature_extractor = Transformer(irreps_output=self.irreps_fragment_features,
        #                                             num_elements=len(self.z_table),
        #                                             max_radius=self.max_radius)
        # self.fragment_feature_extractor = self.fragment_feature_extractor.to(self.device)
        
        self.fragment_features = torch.nn.Parameter(self.irreps_fragment_features.randn(self.n_fragments, -1),
                                                    requires_grad=False).to(self.device)
        
        # Fragment logit + 3D vector 
        self.actor_irreps = o3.Irreps(f'1x0e')
        self.actor = o3.FullyConnectedTensorProduct(irreps_in1=self.irreps_pocket_features[1:], 
                                                    irreps_in2=self.irreps_fragment_features[1:],
                                                    irreps_out=self.actor_irreps,
                                                    internal_weights=True)
        self.actor.to(self.device)
        
        self.critic_irreps = o3.Irreps(f'1x0e')
        self.critic = o3.Linear(irreps_in=self.irreps_pocket_features, 
                                irreps_out=self.critic_irreps)
        self.critic.to(self.device)
        
        # Prepare rotation matrix for the fragment features
        self.torsion_angles_rad = torch.deg2rad(torch.tensor(self.torsion_angles_deg))
        self.rot_mats = o3.matrix_x(self.torsion_angles_rad)
        self.d_irreps_fragment = self.irreps_fragment_features.D_from_matrix(self.rot_mats) # (n_angles, irreps_dim, irreps_dim)
        self.d_irreps_fragment = self.d_irreps_fragment.float().to(self.device)


    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        action = self.get_action(features, masks)
        value = self.get_value(features)
        return action, value


    def extract_features(self,
                         x: Batch):
        features = self.pocket_feature_extractor(x)
        return features
    
    
    def extract_fragment_features(self):
        fragment_embeddings = torch.einsum('bi,rji->brj', self.fragment_features, self.d_irreps_fragment)
        fragment_embeddings = fragment_embeddings.reshape(-1, self.irreps_fragment_features.dim) # (n_fragment*n_rotations, irreps_dim)
        return fragment_embeddings
    

    def get_action(self, 
                   features: torch.Tensor, # (n_pockets, irreps_pocket_features)
                   fragment_features: torch.Tensor,
                   masks: torch.Tensor = None, # (n_pockets, n_fragments)
                   frag_actions: torch.Tensor = None # (action_dim)
                   ) -> Action:
            
        n_pockets = features.shape[0]
        
        # Apply fragment rotations
        n_fragment_rot = self.action_dim
        # [fragment1rot1, fragment1rot2, fragmentnrotn-1, fragmentnrotn] * n_pocket
        fragment_indices = torch.arange(n_fragment_rot)
        fragment_indices = fragment_indices.repeat(n_pockets)
        
        fragment_embeddings = fragment_features[fragment_indices]
        
        # [pocket1, pocket1, pocket1... pocketn, pocketn, pocketn] with each pocket duplicated n_fragment*n_rot times
        pocket_indices = torch.arange(n_pockets)
        pocket_indices = torch.repeat_interleave(pocket_indices, n_fragment_rot)
        pocket_embeddings = features[pocket_indices] # (n_pocket*n_rotations*self.n_fragments, irreps_dim)
        
        # Embedding dimension is
        # Pocket0Frag0Rot0, Pocket0Frag0Rot1, ... Pocket0Frag1Rot0, Pocket0Frag1Rot1... PocketnFragnRotn-1, PocketnFragnRotn
        
        _, pocket_equi_embeddings = self.split_features(pocket_embeddings, irreps=self.irreps_pocket_features)
        _, fragment_equi_embeddings = self.split_features(fragment_embeddings, irreps=self.irreps_fragment_features)
        
        actor_output = self.actor(pocket_equi_embeddings, fragment_equi_embeddings)
            
        actor_output = actor_output.reshape(n_pockets, self.action_dim)
        
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
        critic_output = self.critic(features)
        value = critic_output
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