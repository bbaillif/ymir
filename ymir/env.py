import numpy as np
import torch
import logging
import copy

from rdkit import Chem
from rdkit.Chem import Mol
from ymir.molecule_builder import (add_fragment_to_seed, 
                                   potential_reactions)
from ymir.utils.fragment import get_fragments_from_mol
from ymir.data.featurizer import RDKitFeaturizer
from ymir.params import TORSION_SPACE_STEP
from ymir.data.structure.complex import Complex
from ymir.reward import DockingBatchRewards
from torch_geometric.data import Data, Batch
from ymir.data import Fragment
from typing import Any, NamedTuple
from ymir.metrics.activity import GlideScore, VinaScore, VinaScorer
 
ATOM_NUMBER_PADDING = 1000

class Action(NamedTuple):
    fragment_i: int
    torsion_value: float

class FragmentBuilderEnv():
    
    def __init__(self,
                 protected_fragments: list[Fragment],
                 torsion_space_step: int = TORSION_SPACE_STEP,
                 max_episode_steps: int = 10,
                 valid_action_masks: dict[int, torch.Tensor] = None,
                 ) -> None:
        self.protected_fragments = protected_fragments
        self.torsion_space_step = torsion_space_step
        self.torsion_values = np.arange(-180, 180, self.torsion_space_step)
        self.n_torsions = len(self.torsion_values)
        
        self.n_fragments = len(self.protected_fragments)
        
        possible_actions: list[Action] = []
        for fragment_i in range(self.n_fragments):
            for torsion_value in self.torsion_values:
                action= Action(fragment_i=fragment_i,
                                torsion_value=torsion_value)
                possible_actions.append(action)
        self._action_dim = len(possible_actions)
            
        if valid_action_masks is None:
        
            self.valid_action_masks: dict[int, list[bool]] = {}
            for attach_label_1, d_potential_attach in potential_reactions.items():
                mask = [False for _ in range(self.action_dim)]
                for act_i, action in enumerate(possible_actions):
                    fragment_i = action.fragment_i
                    fragment = self.protected_fragments[fragment_i]
                    attach_points = fragment.get_attach_points()
                    assert len(attach_points) == 1
                    attach_label = list(attach_points.values())[0]
                    if attach_label in d_potential_attach:
                        mask[act_i] = True
                self.valid_action_masks[attach_label_1] = torch.tensor(mask)
            
        else:
            for mask in valid_action_masks.values():
                assert mask.size()[-1] == self.action_dim
            self.valid_action_masks = valid_action_masks
        
        self.max_episode_steps = max_episode_steps
        self.rdkit_featurizer = RDKitFeaturizer()
        self.seed: Fragment = None
        self.fragment: Fragment = None
        
    @property
    def action_dim(self) -> int:
        return self._action_dim
        
        
    def action_to_fragment_build(self,
                                 action: int):
        
        fragment_i = action // self.n_torsions
        try:
            protected_fragment = self.protected_fragments[fragment_i]
        except:
            import pdb;pdb.set_trace()
        torsion_i = action % self.n_torsions
        torsion_value = self.torsion_values[torsion_i]
        
        product = add_fragment_to_seed(seed=self.seed,
                                        fragment=protected_fragment,
                                        torsion_value=torsion_value)
        self.seed = product
        
    
    def _get_obs(self):
        data = self.featurize_pocket()
        obs = data
        return obs
    
    
    def _get_info(self):
        return {'protein_path': self.complex.glide_protein.pdb_filepath,
                'seed_i': self.seed_i}
    
    
    def featurize_pocket(self) -> Data:
        
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
        
        protein_x.extend(ligand_x)
        x = protein_x
        
        protein_pos.extend(ligand_pos)
        pos = protein_pos
        
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        data = Data(x=x,
                    pos=pos)

        return data
    
    
    def reset(self, 
              complx: Complex) -> tuple[Data, dict[str, Any]]:
        
        self.complex = complx
        fragments = get_fragments_from_mol(self.complex.ligand)
        self.seed_i = np.random.choice(len(fragments))
        self.seed = fragments[self.seed_i]
        
        self.original_seed = Chem.Mol(self.seed)
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        self.truncated = False
        
        assert self.terminated == False
        
        # observation = self._get_obs()
        info = self._get_info()
        
        self.actions = []
        
        # return observation, info
        return info
    
    
    def set_focal_atom_id(self) -> bool:
        
        terminated = False
        self.attach_points = self.seed.get_attach_points()
        if len(self.attach_points) == 0:
            terminated = True
            self.focal_atom_id = None
        else:
            self.focal_atom_id = np.random.choice(list(self.attach_points.keys()))
            self.seed.protect(atom_ids_to_keep=[self.focal_atom_id])
        return terminated
    
    
    def step(self,
             action: int):
        
        self.actions.append(action)
        n_actions = len(self.actions)
        
        self.action_to_fragment_build(action) # seed is deprotected
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        
        if n_actions == self.max_episode_steps: # not terminated but reaching max step size
            self.truncated = True
            # We replace the focal atom with hydrogen (= protect)
            # all other attachment points are already protected
            self.seed.protect() 
        
        if self.terminated or self.truncated:
            assert(all([atom.GetAtomicNum() > 0 for atom in self.seed.GetAtoms()]))
            
        reward = 0 # reward is handled in batch, once all envs are done
        
        # observation = self._get_obs()
        info = self._get_info()
        
        # return observation, reward, self.terminated, self.truncated, info
        return reward, self.terminated, self.truncated, info
    
    
    def get_valid_action_mask(self):
        try:
            attach_label = self.attach_points[self.focal_atom_id]
        except:
            import pdb;pdb.set_trace()
        valid_action_mask = self.valid_action_masks[attach_label]
        return valid_action_mask
    
    
    def get_clean_mol(self):
        frag = Fragment(self.seed)
        frag.protect()
        Chem.SanitizeMol(frag)
        mol = Chem.RemoveHs(frag)
        mol_h = Chem.AddHs(mol, addCoords=True)
        return mol_h
    
    
class BatchEnv():
    
    def __init__(self,
                 envs: list[FragmentBuilderEnv]) -> None:
        self.envs = envs
        
        
    def reset(self,
              complx: Complex,
              reward_function: DockingBatchRewards) -> tuple[list[Data], list[dict[str, Any]]]:
        # batch_obs: list[Data] = []
        batch_info: list[dict[str, Any]] = []
        for env in self.envs:
            # obs, info = env.reset(complx)
            info = env.reset(complx)
            # batch_obs.append(obs)
            batch_info.append(info)
        # batch_obs = Batch.from_data_list(batch_obs)
        # return batch_obs, batch_info
        self.terminateds = [False] * len(self.envs)
        self.ongoing_env_idxs = list(range(len(self.envs)))
        
        self.reward_function = reward_function
        
        return batch_info
    
    
    def get_valid_action_mask(self) -> torch.Tensor:
        masks = []
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            valid_action_mask = env.get_valid_action_mask()
            masks.append(valid_action_mask)
        masks = torch.stack(masks)
        non_terminated = [not terminated for terminated in self.terminateds]
        try:
            assert masks.size()[0] == sum(non_terminated)
        except:
            import pdb;pdb.set_trace()
        return masks
    
    
    def get_ongoing_envs(self) -> list[FragmentBuilderEnv]:
        ongoing_envs = [env 
                        for env, terminated in zip(self.envs, self.terminateds)
                        if not terminated]
        return ongoing_envs
    
    
    def step(self,
             actions: torch.Tensor,
             ) -> tuple[list[float], list[bool], list[bool], list[dict]]:
        assert actions.size()[0] == len(self.ongoing_env_idxs)
        # batch_obs: list[Data] = []
        # batch_reward = []
        batch_truncated = []
        batch_info = []
        mols = []
        for env_i, action in zip(self.ongoing_env_idxs, actions):
            env = self.envs[env_i]
            # obs, reward, terminated, truncated, info = env.step(action)
            _, terminated, truncated, info = env.step(action)
            # batch_obs.append(obs)
            # batch_reward.append(reward)
            
            if terminated: # we have no attachment points left
                self.terminateds[env_i] = True
            
            batch_truncated.append(truncated)
            batch_info.append(info)
            
            mols.append(env.get_clean_mol())
            
        batch_rewards = self.reward_function.get_rewards(mols)
        
        for i, (reward, truncated) in enumerate(zip(batch_rewards, batch_truncated)):
            if truncated:
                batch_rewards[i] = -10
        
        # this will cause truncation to extend to termination
        for env_i, reward in zip(self.ongoing_env_idxs, batch_rewards):
            if reward < 0: # there is a clash, or bad interaction, or too many atoms
                self.terminateds[env_i] = True
            
        self.ongoing_env_idxs = [i 
                                 for i, terminated in enumerate(self.terminateds) 
                                 if not terminated]
        logging.debug(self.ongoing_env_idxs)
        logging.debug(self.terminateds)
            
        # batch_obs = Batch.from_data_list(batch_obs)
        # return batch_obs, batch_reward, batch_terminated, batch_truncated, batch_info
        return batch_rewards, list(self.terminateds), batch_truncated, batch_info
    
    
    def get_clean_mols(self) -> list[Mol]:
        
        mols = []
        for env in self.envs:
            mol_h = env.get_clean_mol()
            mols.append(mol_h)
        return mols
            
            
    def get_obs(self,
                all_envs: bool = False) -> list[Data]:
        obs_list = []
        if all_envs:
            env_idxs = range(len(self.envs))
        else: # take ongoing
            env_idxs = self.ongoing_env_idxs
        for env_i in env_idxs:
            env = self.envs[env_i]
            obs = env.featurize_pocket()
            obs_list.append(obs)
        return obs_list
    
    
    def save_state(self) -> None:
        save_path = 'ligands.sdf' 
        with Chem.SDWriter(save_path) as writer:
            for env in self.envs:
                mol_h = env.get_clean_mol()
                writer.write(mol_h)
        with Chem.SDWriter('pocket.sdf') as writer:   
            writer.write(env.complex.pocket.mol)
        with Chem.SDWriter('native_ligand.sdf') as writer:   
            writer.write(env.complex.ligand)