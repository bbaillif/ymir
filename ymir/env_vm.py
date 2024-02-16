import numpy as np
import torch
import logging
import copy

from rdkit import Chem
from rdkit.Chem import Mol
from ymir.molecule_builder import (add_fragment_to_seed, 
                                   potential_reactions)
from ymir.utils.fragment import get_fragments_from_mol
from ymir.data.structure.complex import Complex
from ymir.reward import DockingBatchRewards
from torch_geometric.data import Data, Batch
from ymir.data import Fragment
from typing import Any
from ymir.metrics.activity import GlideScore, VinaScore, VinaScorer


class FragmentBuilderEnv():
    
    def __init__(self,
                 protected_fragments: list[Fragment],
                 valid_action_masks: dict[int, torch.Tensor],
                 max_episode_steps: int = 10,
                 ) -> None:
        self.protected_fragments = protected_fragments
        
        self.n_fragments = len(self.protected_fragments)
        
        for mask in valid_action_masks.values():
            assert mask.size()[-1] == self.action_dim
        self.valid_action_masks = valid_action_masks
        
        self.max_episode_steps = max_episode_steps
        self.seed: Fragment = None
        self.fragment: Fragment = None
        
    @property
    def action_dim(self) -> int:
        return self.n_fragments
        
        
    def action_to_fragment_build(self,
                                 frag_action: int,
                                 angle_action: float):
        
        protected_fragment = self.protected_fragments[frag_action]
        
        torsion_value = np.rad2deg(angle_action)
        product = add_fragment_to_seed(seed=self.seed,
                                        fragment=protected_fragment,
                                        torsion_value=torsion_value)
        self.seed = product
        
    
    def _get_obs(self):
        data = self.featurize_pocket()
        obs = data
        return obs
    
    
    def _get_info(self):
        return {'seed_i': self.seed_i}
    
    
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
            assert len(attach_x) == 1
            assert len(attach_pos) == 1
        
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
             frag_action: int,
             angle_action: float):
        
        self.actions.append((frag_action, angle_action))
        n_actions = len(self.actions)
        
        self.action_to_fragment_build(frag_action, angle_action) # seed is deprotected
        
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
        attach_label = self.attach_points[self.focal_atom_id]
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
              complx: Complex) -> tuple[list[Data], list[dict[str, Any]]]:
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
        
        # self.scorer = GlideScore(glide_protein=complx.glide_protein)
        vina_scorer = VinaScorer(complx.vina_protein)
        vina_scorer.set_box_from_ligand(complx.ligand)
        scorer = VinaScore(vina_scorer=vina_scorer)
        self.reward_function = DockingBatchRewards(complx, scorer)
        
        return batch_info
    
    
    def get_valid_action_mask(self) -> torch.Tensor:
        masks = []
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            valid_action_mask = env.get_valid_action_mask()
            masks.append(valid_action_mask)
        masks = torch.stack(masks)
        non_terminated = [not terminated for terminated in self.terminateds]
        
        assert masks.size()[0] == sum(non_terminated)
        
        return masks
    
    
    def get_ongoing_envs(self) -> list[FragmentBuilderEnv]:
        ongoing_envs = [env 
                        for env, terminated in zip(self.envs, self.terminateds)
                        if not terminated]
        return ongoing_envs
    
    
    def step(self,
             frag_actions: torch.Tensor,
             angle_actions: torch.Tensor,
             ) -> tuple[list[float], list[bool], list[bool], list[dict]]:
        assert frag_actions.size()[0] == len(self.ongoing_env_idxs)
        assert angle_actions.size()[0] == len(self.ongoing_env_idxs)
        batch_truncated = []
        batch_info = []
        mols = []
        for env_i, frag_action, angle_action in zip(self.ongoing_env_idxs, frag_actions, angle_actions):
            env = self.envs[env_i]
            aa = angle_action[frag_action]
            frag_action = frag_action.cpu()
            aa = aa.cpu()
            _, terminated, truncated, info = env.step(frag_action, aa)
            
            if terminated: # we have no attachment points left
                self.terminateds[env_i] = True
            
            batch_truncated.append(truncated)
            batch_info.append(info)
            
            mols.append(env.get_clean_mol())
            
        batch_rewards = self.reward_function.get_rewards(mols, self.terminateds)
        
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