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
from ymir.featurizer import Featurizer
    
    
class Agent(nn.Module):
    
    def __init__(self, 
                 fragments: list[Fragment],
                 z_table: AtomicNumberTable,
                 hidden_irreps: o3.Irreps,
                 r_max: float = 5.0,
                 device: torch.device = torch.device('cuda')
                 ):
        super(Agent, self).__init__()
        
        # self.fragments = fragments
        self.n_fragments = len(fragments)
        self.z_table = z_table
        self.device = device
        
        self.embedding_extractor = MACE(hidden_irreps=hidden_irreps,
                                      num_elements=len(self.z_table),
                                      r_max=r_max)
        self.embedding_extractor = self.embedding_extractor.to(device)
        
        self.featurizer = Featurizer(z_table=self.z_table)
        self.fragment_features = []
        for fragment in fragments:
            fragment_x = []
            fragment_pos = []
            attach_x = []
            attach_pos = []
            fragment_positions = fragment.GetConformer().GetPositions()
            for atom_i, atom in enumerate(fragment.GetAtoms()):
                atomic_num = atom.GetAtomicNum()
                atom_pos = fragment_positions[atom_i]
                feature = [atomic_num]
                if atomic_num == 0:
                    attach_x.append(feature)
                    attach_pos.append(atom_pos.tolist())
                else:
                    fragment_x.append(feature)
                    fragment_pos.append(atom_pos.tolist())
            fragment_x.extend(attach_x) # make sure the attach environment is last
            fragment_pos.extend(attach_pos)
            data = self.featurizer.featurize_mol(fragment_x, fragment_pos)
            self.fragment_features.append(data)
        self.fragment_features = Batch.from_data_list(self.fragment_features)
        self.fragment_features.to(self.device)
        
        self.actor = o3.FullyConnectedTensorProduct(irreps_in1=hidden_irreps,
                                                 irreps_in2=hidden_irreps,
                                                 irreps_out='1x0e',
                                                 internal_weights=True,
                                                 shared_weights=True)
        
        self.critic = Linear(in_features=hidden_irreps.dim,
                            out_features=1,
                            device=device)
        
        self.actor.to(device)
        self.critic.to(device)

    def forward(self, 
                x,
                masks) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.extract_embeddings(x)
        actions, logprobs, entropies = self.get_action(embeddings, 
                                                        masks)
        value = self.get_value(embeddings)
        return actions, value


    def extract_embeddings(self,
                         x: Batch):
        embeddings = self.embedding_extractor(x)
        return embeddings
    

    def get_action(self, 
                   pocket_embeddings: torch.Tensor, 
                   masks: torch.Tensor = None,
                   actions: torch.Tensor = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        fragment_embeddings = self.embedding_extractor(self.fragment_features)
        n_pockets = pocket_embeddings.shape[0]
        
        # Repeat the vectors such as shape[0] is (n_pockets, n_fragments)
        # Since the Tensorproduct does not work with 3 dimensions
        # [pocket1, pocket2, pocket3] * n_fragment
        pocket_embeddings = torch.repeat_interleave(pocket_embeddings, self.n_fragments, dim=0)
        
        # [fragment1, fragment1, fragment1,..., fragmentn, fragmentn, fragmentn]
        fragment_embeddings = fragment_embeddings.repeat(n_pockets, 1)
        
        output = self.actor(pocket_embeddings, fragment_embeddings)
        logits = output.reshape(n_pockets, self.n_fragments)
        
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
                  pocket_embeddings: torch.Tensor) -> torch.Tensor:
        # inv_features, equi_features = features.split([self.irreps_scalars_dim, self.irreps_vectors_dim], dim=-1)
        # value = self.critic(inv_features)
        value = self.critic(pocket_embeddings)
        return value