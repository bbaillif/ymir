import gymnasium as gym
import numpy as np
import torch
import time

from gymnasium.spaces import Box, GraphInstance, Sequence
from rdkit import Chem
from rdkit.Chem import Mol
from ymir.old.molecule_builder import add_fragment_to_seed
from ymir.utils.fragment3d import (get_fragments_from_mol, 
                             get_attach_points, 
                             protect_fragment, 
                             get_neighbor_id_for_atom_id)
from ymir.data.featurizer import RDKitFeaturizer, MDAFeaturizer
from MDAnalysis import Universe
from ymir.params import TORSION_SPACE_STEP
from ymir.data.structure.complex import Complex
from rdkit.Chem import GetPeriodicTable
from torch_geometric.data import Data
from ymir.metrics.activity import GlideScore
from rdkit.Chem.rdMolAlign import AlignMol

ATOM_NUMBER_PADDING = 1000

class FragmentBuilderEnv(gym.Env):
    
    def __init__(self,
                 fragments: list[Mol],
                 fragments_protections: list[dict[int, int]],
                 complexes: list[Complex],
                 valid_action_masks: dict[int, list[bool]],
                 action_dim: int,
                 node_dim: int,
                 edge_dim: int = 0,
                 torsion_space_step = TORSION_SPACE_STEP,
                 max_episode_steps = 10,
                 ) -> None:
        
        self.node_dim = node_dim
        self.node_space = Box(low=-np.inf, high=np.inf, shape=(ATOM_NUMBER_PADDING, node_dim)) # padding to a large number of atom, I hate this
        # self.pos_space = Box(low=-np.inf, high=np.inf, shape=(3, ))
        # self.edge_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(edge_dim,))
        # self.observation_space = gym.spaces.Graph(node_space=self.node_space,
        #                                           edge_space=None)
        
        self.observation_space = self.node_space
        
        # self.observation_space = Dict({'x': Sequence(self.node_space),
        #                                'pos': Sequence(self.pos_space)})
        # self.observation_space = None
        
        self.complexes = complexes
        self.n_complexes = len(self.complexes)
        
        self.valid_action_masks = valid_action_masks
        
        self.action_space = gym.spaces.Discrete(action_dim)
        self.fragments = fragments
        self.fragments_protections = fragments_protections
        self.torsion_space_step = torsion_space_step
        self.torsion_values = np.arange(-180, 180, self.torsion_space_step)
        self.n_torsions = len(self.torsion_values)
        
        self.max_episode_steps = max_episode_steps
        # self.rdkit_featurizer = RDKitFeaturizer()
        # self.mda_featurizer = MDAFeaturizer()
        
        
    def action_to_fragment_build(self,
                                 action: int):
        
        fragment_i = action // self.n_torsions
        try:
            fragment = self.fragments[fragment_i]
        except:
            import pdb;pdb.set_trace()
        fragment_protections = self.fragments_protections[fragment_i]
        torsion_i = action % self.n_torsions
        torsion_value = self.torsion_values[torsion_i]
        
        seed_attach_atom_ids = list(get_attach_points(self.seed).keys())
        fragment_attach_atom_ids = list(get_attach_points(fragment).keys())
        assert len(seed_attach_atom_ids) == 1
        assert len(fragment_attach_atom_ids) == 1
        
        atom_id1 = seed_attach_atom_ids[0]
        neighbor_id1 = get_neighbor_id_for_atom_id(mol=self.seed, atom_id=atom_id1)
        
        atom_id2 = list(get_attach_points(fragment).keys())[0]
        neighbor_id2 = get_neighbor_id_for_atom_id(mol=fragment, atom_id=atom_id2)
        
        rmsd = AlignMol(fragment, self.seed, atomMap=[(atom_id2, neighbor_id1), (neighbor_id2, atom_id1)])
        
        product = add_fragment_to_seed(seed=self.seed,
                                        seed_protections=self.seed_protections,
                                        fragment=fragment,
                                        fragment_protections=fragment_protections,
                                        torsion_value=torsion_value)
        self.seed = product
        
    
    def _get_obs(self):
        obs = self.featurize_pocket()
        n_atoms = obs.shape[0]
        assert n_atoms < ATOM_NUMBER_PADDING, print(f'n_atoms is too large: {n_atoms}')
        n_padding = ATOM_NUMBER_PADDING - n_atoms
        padding = np.empty((n_padding, self.node_dim))
        padding[:] = np.nan
        obs = np.concatenate([obs, padding], axis=0)
        return obs
    
    
    def _get_info(self):
        return {'protein_path': self.complex.glide_protein.pdb_filepath,
                'seed_i': self.seed_i}
    
    
    def featurize_pocket(self):
        
        ligand_x = []
        ligand_pos = []
        attach_x = []
        attach_pos = []
        ligand_positions = self.seed.GetConformer().GetPositions()
        for atom_i, atom in enumerate(self.seed.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            atom_pos = ligand_positions[atom_i]
            feature = [atomic_num]
            if atomic_num == 0:
                attach_x.append(feature)
                attach_pos.append(atom_pos.tolist())
            else:
                ligand_x.append(feature)
                ligand_pos.append(atom_pos.tolist())
                
        if not (self.terminated or self.truncated): # There must be an attach point, it will be the last value
            try:
                assert len(attach_x) == 1
                assert len(attach_pos) == 1
            except Exception as e:
                import pdb;pdb.set_trace()
        
            ligand_x.extend(attach_x) # make sure the attach environment is last
            ligand_pos.extend(attach_pos)
        
        pocket_mol: Mol = self.complex.pocket.mol
        protein_x = [[atom.GetAtomicNum()] for atom in pocket_mol.GetAtoms()]
        protein_pos = pocket_mol.GetConformer().GetPositions()
        protein_pos = protein_pos.tolist()
        
        # protein_atoms = self.complex.pocket.mol.atoms
        # protein_pos: list = protein_atoms.positions.tolist()
        # protein_elements = [atom.element for atom in protein_atoms]
        # periodic_table = GetPeriodicTable()
        # protein_x = [[periodic_table.GetAtomicNumber(element)] for element in protein_elements]
        
        protein_x.extend(ligand_x)
        x = protein_x
        
        protein_pos.extend(ligand_pos)
        pos = protein_pos
        
        # data = Data(x=x, 
        #             pos=pos)
        
        # x = torch.Tensor(x, dtype=torch.float32)
        # pos = torch.Tensor(pos, dtype=torch.float32)
        # features = torch.cat([x, pos], dim=1)
        x = np.array(x)
        pos = np.array(pos)
        features = np.concatenate([x, pos], axis=1)
        
        assert features.shape[1] == self.node_dim
        
        # data = GraphInstance(nodes=features)
        data = features
        
        # ligand_data_list = self.rdkit_featurizer.featurize_mol(self.seed)
        # ligand_data = ligand_data_list[0]
        # data = ligand_data + self.complex.protein_pocket_data # FIND A WAY TO MERGE EXISTING PROTEIN DATA WITH LIGAND DATA
        return data
    
    
    def reset(self, 
              seed=None, 
              options=None):
        
        super().reset(seed=seed)
        
        n_frag = 0
        while n_frag <= 1:
            self.complx_i = self.np_random.choice(self.n_complexes)
            self.complex = self.complexes[self.complx_i]
            fragments = get_fragments_from_mol(self.complex.ligand)
            n_frag = len(fragments)
        self.seed_i = self.np_random.choice(len(fragments))
        self.seed = fragments[self.seed_i]
        
        self.original_seed = Chem.Mol(self.seed)
        
        self.terminated = self.set_focal_point() # seed is protected
        self.truncated = False
        
        assert self.terminated == False
        
        observation = self._get_obs()
        info = self._get_info()
        
        self.actions = []
        
        return observation, info
    
    
    def set_focal_point(self) -> bool:
        
        terminated = False
        self.attach_points = get_attach_points(mol=self.seed)
        if len(self.attach_points) == 0:
            terminated = True
            self.focal_point = None
            self.seed_protections = None
        else:
            self.focal_point = self.np_random.choice(list(self.attach_points.keys()))
            self.seed_protections = protect_fragment(mol=self.seed, 
                                                    atom_ids_to_keep=[self.focal_point])
        return terminated
    
    
    def step(self,
             action: int):
        
        self.actions.append(action)
        n_actions = len(self.actions)
        
        self.action_to_fragment_build(action) # seed is deprotected
        
        self.terminated = self.set_focal_point() # seed is protected
        
        if n_actions == self.max_episode_steps: # not terminated but reaching max step size
            # We replace the focal atom with hydrogen (= protect)
            # All other attachment points are already protected
            self.truncated = True
            _ = protect_fragment(mol=self.seed)
        
        if self.terminated or self.truncated:
            assert(all([atom.GetAtomicNum() > 0 for atom in self.seed.GetAtoms()]))
            Chem.SanitizeMol(self.seed)
            self.seed_h = Chem.AddHs(self.seed, addCoords=True)
            reward = self.get_reward()

            # with Chem.SDWriter('ligand.sdf') as writer:
            #     writer.write(self.seed_h)
                
            # with Chem.SDWriter('pocket.sdf') as writer:
            #     writer.write(self.complex.pocket.mol)
                
            # import pdb;pdb.set_trace()
        else:
            # print('New step done')
            reward = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, self.terminated, self.truncated, info
        
        
    def get_reward(self):
        
        t0 = time.time()
        glide_scores = self.get_glide_score()
        # glide_scores = [1]
        glide_score = glide_scores[0]
        reward = - glide_score # we want to maximize the reward, so minimize glide score
        # / np.sqrt(self.seed.GetNumAtoms())
        # INCLUDE CLASH SCORE ?
        # INCLUDE TORSION PREFERENCE PENALTY ?
        t1 = time.time()
        print('Time for reward: ', t1 - t0)
        return reward
    
    
    def get_glide_score(self):
        glide_protein = self.complex.glide_protein
        glide_scorer = GlideScore(glide_protein)
        glide_score = glide_scorer.score_mol(self.seed_h)
        return glide_score
    
    
    def get_valid_action_mask(self):
        attach_label = self.attach_points[self.focal_point]
        valid_action_mask = self.valid_action_masks[attach_label]
        return valid_action_mask