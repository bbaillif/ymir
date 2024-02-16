import torch
import logging
import os

from torch.nn import Module, Sequential, Linear, ReLU

from torch import nn
from ymir.params import FEATURES_DIM
from torch_geometric.data import Batch
from e3nn import o3
from ymir.model import MACE
from ymir.data import Fragment
from ymir.atom_environment import SHEnvironment, AtomicNumberTable
from ymir.distribution import CategoricalMasked
from tqdm import tqdm
from ymir.params import LMAX
    
    
class Agent(nn.Module):
    
    def __init__(self, 
                 fragments: list[Fragment],
                 z_table: AtomicNumberTable,
                 lmax: int = LMAX,
                 device: torch.device = torch.device('cuda')
                 ):
        super(Agent, self).__init__()
        
        # self.fragments = fragments
        self.n_fragments = len(fragments)
        self.z_table = z_table
        self.device = device
        
        self.sh_environment = SHEnvironment(z_table, 
                                            lmax,
                                            device=device)
        attach_points = [fragment.get_attach_points() 
                         for fragment in fragments]
        attach_atom_ids = [list(d.keys())[0] for d in attach_points]
        
        fragment_envs_path = '/home/bb596/hdd/ymir/fragments_envs.pt'
        if not os.path.exists(fragment_envs_path):
            self.fragment_environments = []
            for fragment, attach_atom_id in tqdm(zip(fragments, attach_atom_ids),
                                                total=len(fragments)):
                fragment.unprotect()
                fragment_environment = self.sh_environment.get_environment(mol=fragment,
                                                                        atom_id=attach_atom_id)
                self.fragment_environments.append(fragment_environment)
            self.fragment_environments = torch.stack(self.fragment_environments) # (n_frag, n_z, n_sh)
            torch.save(self.fragment_environments, fragment_envs_path)
        else:
            self.fragment_environments = torch.load(fragment_envs_path)
            
        self.fragment_environments = self.fragment_environments.to(device)
        
        assert self.n_fragments == self.fragment_environments.shape[0], \
            'The number of input fragments should be the same as the number of frag envs'
        
        # self.feature_extractor = MACE(hidden_irreps=hidden_irreps,
        #                               num_elements=num_elements)
        # self.feature_extractor = self.feature_extractor.to(device)
        
        # self.actor = Linear(in_features=self.n_fragments,
        #                     out_features=self.n_fragments,
        #                     device=device)
        self.critic = Linear(in_features=self.n_fragments,
                            out_features=1,
                            device=device)
        
        self.tp = o3.FullyConnectedTensorProduct(irreps_in1='13x0e + 13x1o + 13x2e + 13x3o',
                                                 irreps_in2='13x0e + 13x1o + 13x2e + 13x3o',
                                                 irreps_out='1x0e',
                                                 internal_weights=True,
                                                 shared_weights=True)
        self.tp = self.tp.to(device)
        
        # self.tp_weights = torch.nn.Parameter(data=torch.Tensor(self.tp.weight_numel), 
        #                                      requires_grad=True)
        
        # self.tp_weights.data.uniform_(-1, 1)

    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        actions, logprobs, entropies = self.get_action(features, 
                                  masks)
        value = self.get_value(features)
        return actions, value


    def extract_features(self,
                         atom_environment: torch.Tensor # (n_env, n_z, n_sh)
                         ) -> torch.Tensor:
        
        # features = torch.tensordot(atom_environment, 
        #                            self.fragment_environments, 
        #                            dims=([-2,-1], [-2,-1]))
        input1 = atom_environment.T.reshape(atom_environment.shape[0], -1) # (n_envs, n_sh * n_z)
        input2 = self.fragment_environments.transpose(-1, -2).reshape(self.fragment_environments.shape[0], -1) # (n_frags, n_sh * n_z)
        
        input1_rep = torch.repeat_interleave(input1, input2.shape[0], dim=0)
        input2_rep = input2.repeat(input1.shape[0], 1)
        
        # import pdb;pdb.set_trace()
        
        features = self.tp(input1_rep, input2_rep) # (n_env, n_frags)
        features = features.reshape(input1.shape[0], input2.shape[0])
        
        return features
    

    def get_action(self, 
                   features: torch.Tensor, 
                   masks: torch.Tensor = None,
                   actions: torch.Tensor = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # logits = self.actor(features)
        logits = features
        
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