import numpy as np
import torch
import logging
import io

from rdkit import Chem
from rdkit.Chem import Mol
from ase.io import read
from ymir.molecule_builder import add_fragment_to_seed
from ymir.utils.fragment import get_fragments_from_mol, get_neighbor_symbol, center_fragment
from ymir.data.structure.complex import Complex
from torch_geometric.data import Data
from ymir.data import Fragment
from typing import Any
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean
from ymir.utils.spatial import rotate_conformer, translate_conformer
from ymir.geometry.geometry_extractor import GeometryExtractor
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist
from ymir.atomic_num_table import AtomicNumberTable
from ymir.params import EMBED_HYDROGENS
from ymir.featurizer_sn import Featurizer
from ymir.metrics.activity import VinaScore
from ymir.bond_distance import MedianBondDistance
from ymir.metrics.activity.vina_cli import VinaCLI
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize


class FragmentBuilderEnv():
    
    def __init__(self,
                 protected_fragments: list[Fragment],
                 z_table: AtomicNumberTable,
                 max_episode_steps: int = 10,
                 valid_action_masks: dict[int, torch.Tensor] = None,
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 ) -> None:
        self.protected_fragments = protected_fragments
        self.z_table = z_table
        self.max_episode_steps = max_episode_steps
        self.valid_action_masks = valid_action_masks
        self.embed_hydrogens = embed_hydrogens
        
        self.n_fragments = len(self.protected_fragments)
        self._action_dim = self.n_fragments
        
        for mask in self.valid_action_masks.values():
            assert mask.size()[-1] == self.n_fragments

        self.seed: Fragment = None
        self.fragment: Fragment = None
        
        self.geometry_extractor = GeometryExtractor()
        self.featurizer = Featurizer(z_table=self.z_table)
        self.mbd = MedianBondDistance()
        
        species = ['C', 'N', 'O', 'Cl', 'S']
        if self.embed_hydrogens:
            species = ['H'] + species
        self.soap = SOAP(species=species,
                        periodic=False,
                        r_cut=8,
                        n_max=8,
                        l_max=6)
        
    @property
    def action_dim(self) -> int:
        return self._action_dim
        
        
    def seed_to_frame(self):

        # Chem.MolToMolFile(self.seed, 'seed_before.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_before.mol')
        
        # Align the neighbor ---> attach point vector to the x axis: (0,0,0) ---> (1,0,0)
        # Then translate such that the neighbor is (0,0,0)
        transformations = center_fragment(self.seed,
                                          attach_to_neighbor=False,
                                          neighbor_is_zero=True)
        
        rotation, translation = transformations
        
        self.transformations.extend(transformations)
        
        # Chem.MolToMolFile(self.seed, 'seed_after.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_after.mol')
        
        
    def get_seed_fragment_distance(self,
                                   fragment: Fragment):
        # Get fragment attach neighbor (which is the seed attach point)
        # Get Seed attach neighbor (which is the fragment attach point)
        # Compute the ideal length between the two atoms
        
        construct_neighbor_symbol = get_neighbor_symbol(self.seed)
        fragment_neighbor_symbol = get_neighbor_symbol(fragment)
        distance = self.mbd.get_mbd(construct_neighbor_symbol, fragment_neighbor_symbol)
        return distance # hard coded but should be modified
        
        
    def get_new_fragments(self,
                        frag_action: int) -> list[Fragment]:
        
        fragment_i = frag_action
        protected_fragment = self.protected_fragments[fragment_i]
        
        new_fragments = [protected_fragment]
        
        return new_fragments
        
        
    def translate_seed(self,
                       fragment: Fragment):
        # translate the seed that had the neighbor at 0 to the neighbor at -interatomic_distance
        # Such that the fragment neighbor that is also the seed attach point is (0,0,0)
        distance = self.get_seed_fragment_distance(fragment) # hard coded but should be modified
        
        translation = np.array([-distance, 0, 0])
        translate_conformer(self.seed.GetConformer(), translation=translation)
        self.transformations.append(translation)
        
        # Chem.MolToMolFile(self.seed, 'seed_after.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_after.mol')
        # import pdb;pdb.set_trace()
        
        
    def action_to_fragment_build(self,
                                 frag_action: int) -> None:
        
        new_fragments = self.get_new_fragments(frag_action)
        self.translate_seed(fragment=new_fragments[0])
        
        new_fragment = new_fragments[0]
        product = add_fragment_to_seed(seed=self.seed,
                                        fragment=new_fragment)
            
        # TODO: put the optimized pose
        self.seed = product
        
    
    def _get_obs(self):
        data = self.featurize_pocket()
        obs = data
        return obs
    
    
    def _get_info(self):
        return {}
    
    
    def rdkit_to_ase(self,
                     mol: Mol):
        filename = 'mol.xyz'
        Chem.MolToXYZFile(mol, filename)
        ase_atoms = read(filename)
        return ase_atoms
    
    
    def reset(self, 
              complx: Complex,
              seed: Fragment,
              scorer: VinaScore,
              seed_i: int,
            #   memory: dict
              ) -> tuple[Data, dict[str, Any]]:
        
        self.complex = complx
        self.seed = Fragment(seed,
                             protections=seed.protections)
        self.scorer = scorer
        self.seed_i = seed_i
        # self.memory = memory
        
        self.transformations = []
        
        random_rotation = Rotation.random()
        rotate_conformer(self.seed.GetConformer(), rotation=random_rotation)
        self.transformations.append(random_rotation)
        
        # self.original_seed = Chem.Mol(self.seed)
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        assert self.terminated is False, 'No attachement point in the seed'
        self.truncated = False
        
        assert not self.terminated
        
        self.seed_to_frame()
        
        # observation = self._get_obs()
        info = self._get_info()
        
        self.actions = []
        
        self.valid_action_mask = self.get_valid_action_mask()
        has_valid_action = torch.any(self.valid_action_mask)
        self.terminated = not has_valid_action
        
        # this should not happen, unless the set of fragments is 
        # not suitable for the given starting situation
        if self.terminated: 
            import pdb;pdb.set_trace()
        
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
             frag_action: int):
        
        self.actions.append(frag_action)
        n_actions = len(self.actions)
        
        self.action_to_fragment_build(frag_action) # seed is deprotected
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        
        # REMOVE FOR MULTI-STEP RL
        if not self.terminated:
            import pdb;pdb.set_trace()
        assert self.terminated, 'Something is wrong with the fragmentation code'
        reward = 0
        
        if n_actions == self.max_episode_steps: # not terminated but reaching max step size
            self.truncated = True
            # We replace the focal atom with hydrogen (= protect)
            # all other attachment points are already protected
            self.seed.protect() 
        elif not self.terminated:
            self.seed_to_frame()
            self.valid_action_mask = torch.tensor(self.get_valid_action_mask())
            
            # we terminate the generation if there is no valid action (due to clash)
            has_valid_action = torch.any(self.valid_action_mask)
            self.terminated = not has_valid_action
            if self.terminated:
                logging.info('We have an environment with only clashing fragments')
                self.seed.protect()
                reward = -100 # it is actually a clash penalty
        
        if self.terminated or self.truncated:
            assert(all([atom.GetAtomicNum() > 0 for atom in self.seed.GetAtoms()]))
        
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
        
        # has_clashes = self.get_clashes() # Takes 400 it/s
        # has_clashes = torch.tensor(has_clashes)
        # valid_positions = torch.logical_not(has_clashes)
        # valid_action_mask = torch.logical_and(valid_action_mask, valid_positions)
        
        return valid_action_mask
    
    
    def get_clean_mol(self,
                      ligand: Mol):
        frag = Fragment(ligand)
        frag.protect()
        Chem.SanitizeMol(frag)
        mol = Chem.RemoveHs(frag)
        mol_h = Chem.AddHs(mol, addCoords=True)
        
        for transformation in reversed(self.transformations):
            if isinstance(transformation, Rotation):
                rotation_inv = transformation.inv()
                rotate_conformer(mol_h.GetConformer(), rotation=rotation_inv)
            elif isinstance(transformation, np.ndarray):
                translation_inv = -transformation
                translate_conformer(mol_h.GetConformer(), translation=translation_inv)
            else:
                import pdb;pdb.set_trace()
        
        return mol_h
    
    
class BatchEnv():
    
    def __init__(self,
                 envs: list[FragmentBuilderEnv],
                 memory: dict) -> None:
        self.envs = envs
        self.memory = memory
        
        
    def reset(self,
              complexes: list[Complex],
              seeds: list[Fragment],
              initial_scores: list[float],
              absolute_scores: list[float] = None,
              scorer: VinaScore = None,
              seed_i: int = None) -> tuple[list[Data], list[dict[str, Any]]]:
        # batch_obs: list[Data] = []
        assert len(complexes) == len(self.envs)
        batch_info: list[dict[str, Any]] = []
        for env, seed, complx in zip(self.envs, seeds, complexes):
            info = env.reset(complx,
                             seed,
                             scorer,
                             seed_i,
                            #  self.memory
                             )
            batch_info.append(info)
        self.terminateds = [False] * len(self.envs)
        self.ongoing_env_idxs = list(range(len(self.envs)))
        
        # vina_cli = VinaCLI()
        # receptor_paths = [env.complex.vina_protein.pdbqt_filepath 
        #                   for env in self.envs]
        # native_ligands = [env.complex.ligand 
        #                   for env in self.envs]
        # native_ligands_h = [Chem.AddHs(mol, addCoords=True) 
        #                     for mol in native_ligands]
        # scores = vina_cli.get(receptor_paths=receptor_paths,
        #                     native_ligands=native_ligands,
        #                     ligands=native_ligands_h)
        
        self.current_scores = initial_scores
        self.absolute_scores = absolute_scores
        
        return batch_info
    
    
    def get_valid_action_mask(self) -> torch.Tensor:
        masks = []
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            valid_action_mask = env.valid_action_mask # the get_valid_action_mask() is called earlier
            masks.append(valid_action_mask)
        masks = torch.stack(masks)
        non_terminated = [not terminated for terminated in self.terminateds]
        # try:
        #     assert masks.size()[0] == sum(non_terminated)
        # except:
        #     import pdb;pdb.set_trace()
        return masks
    
    
    def get_ongoing_envs(self) -> list[FragmentBuilderEnv]:
        ongoing_envs = [env 
                        for env, terminated in zip(self.envs, self.terminateds)
                        if not terminated]
        return ongoing_envs
    
    
    def get_rewards(self,
                    max_num_heavy_atoms: int = 50,
                    clash_penalty: float = -10.0) -> list[float]:
        
        rewards = []
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            terminated = self.terminateds[env_i]
            mol = env.get_clean_mol(env.seed)
            # n_clashes = env.complex.clash_detector.get([mol])
            # n_clash = n_clashes[0]
            has_clash = env.get_pocket_ligand_clash(ligand=env.seed)
            if has_clash: # we penalize clash
                reward = clash_penalty
            elif terminated: # if no clash, we score the ligand is construction is finished
                # scores = env.vina_score.get([mol])
                scores = self.scorer.get([mol])
                score = scores[0]
                reward = - score # we want to maximize the reward, so minimize glide score
                reward = reward * reward # we square the score to give higher importance to high scores
                if score > 0: # penalize positive vina score
                    reward = - reward
            else: # construction is ongoing
                reward = 0
                
            # Penality if more than X heavy atoms
            n_atoms = mol.GetNumHeavyAtoms()
            if n_atoms > max_num_heavy_atoms:
                malus = n_atoms - max_num_heavy_atoms
                reward = reward - malus
            
            rewards.append(reward)
            
        assert len(rewards) == len(self.ongoing_env_idxs)
        return rewards
            
    
    def step(self,
             frag_actions: torch.Tensor,
             ) -> tuple[list[float], list[bool], list[bool], list[dict]]:
        
        assert frag_actions.size()[0] == len(self.ongoing_env_idxs)

        batch_truncated = []
        batch_info = []
        mols = []
        old_batch_rewards = []
        for env_i, frag_action in zip(self.ongoing_env_idxs, frag_actions):
            env = self.envs[env_i]
            frag_action = int(frag_action)
            reward, terminated, truncated, info = env.step(frag_action)
            
            if terminated: # we have no attachment points left
                self.terminateds[env_i] = True
            
            old_batch_rewards.append(reward)
            batch_truncated.append(truncated)
            batch_info.append(info)
            
            mols.append(env.get_clean_mol(env.seed))
        
        # import time
        # st = time.time()
        
        # Compute Vina score
        if self.absolute_scores is None:
        
            vina_cli = VinaCLI()
            receptor_paths = [env.complex.vina_protein.pdbqt_filepath 
                            for env in self.envs]
            native_ligands = [env.complex.ligand 
                            for env in self.envs]
            scores = vina_cli.get(receptor_paths=receptor_paths,
                                native_ligands=native_ligands,
                                ligands=mols)
            
        # Take already computed scores
        else:
            masks = self.get_valid_action_mask()
            scores = []
            for mask, absolute_scores, action_tensor in zip(masks, self.absolute_scores, frag_actions):
                
                try:
                    non_zero_idx = torch.nonzero(mask).squeeze() # get list of index with valid action
                    valid_idx = (non_zero_idx == action_tensor.item()).nonzero(as_tuple=True)[0]
                
                # d = {}
                # n = 0
                # for i, value in enumerate(mask):
                #     if value:
                #         d[i] = n
                #         n += 1
                
                    # valid_idx = d[action_tensor.item()]
                    score = absolute_scores[valid_idx]
                    scores.append(score)
                    self.absolute_scores = None # Only works for the first step, was not made for RL
                except Exception as e:
                    print(e)
                    import pdb;pdb.set_trace()
            
                # valid_idx = [d[i.item()] for i in frag_actions]
                # scores = [self.absolute_scores[i] for i in valid_idx]
        
        relative_scores = [new_score - previous_score 
                        for new_score, previous_score in zip(scores, self.current_scores)]
        batch_rewards = [-score for score in relative_scores]
        
        self.current_scores = scores
        
        # print(batch_rewards)
        # print(time.time() - st)
        # import pdb;pdb.set_trace()
        
        
        for i, (reward, truncated) in enumerate(zip(batch_rewards, batch_truncated)):
            if truncated:
                batch_rewards[i] = -1.0
        
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
            mol_h = env.get_clean_mol(env.seed)
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
        with Chem.SDWriter(save_path) as ligand_writer:
            for env in self.envs:
                mol_h = env.get_clean_mol(env.seed)
                ligand_writer.write(mol_h)
              
        pocket_path = 'pockets.sdf'  
        with Chem.SDWriter(pocket_path) as pocket_writer:
            for env in self.envs:
                clean_pocket = env.get_clean_pocket()
                pocket_writer.write(clean_pocket)
                
        native_ligand_path = 'native_ligands.sdf'
        with Chem.SDWriter(native_ligand_path) as native_writer:
            for env in self.envs:
                native_ligand = env.complex.ligand
                native_writer.write(native_ligand)
                
        # with Chem.SDWriter('pocket.sdf') as writer:   
        #     writer.write(env.complex.pocket.mol)
        # with Chem.SDWriter('native_ligand.sdf') as writer:   
        #     writer.write(env.complex.ligand)