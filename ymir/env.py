import numpy as np
import torch
import logging
import io
import os

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
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate
from rdkit.Geometry import Point3D
from ymir.pdbqt_reader import PDBQTReader
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from scipy.spatial.distance import euclidean
from ymir.complex_minimizer import ComplexMinimizer


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
        
    @property
    def action_dim(self) -> int:
        return self._action_dim
        
        
    def reset(self, 
              complx: Complex,
              seed: Fragment,
              initial_score: float,
            #   memory: dict
              ) -> tuple[Data, dict[str, Any]]:
        
        self.complex = complx
        self.seed = Fragment(seed,
                             protections=seed.protections)
        # self.memory = memory
        self.initial_score = initial_score
        self.current_score = initial_score
        
        self.pocket_mol = Mol(self.complex.pocket.mol)
        
        self.transformations = []
        
        random_rotation = Rotation.random()
        rotate_conformer(self.pocket_mol.GetConformer(), rotation=random_rotation)
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
        
        self.complex_minimizer = ComplexMinimizer(pocket=self.complex.pocket)
        
        # return observation, info
        return info
        
        
    def seed_to_frame(self):

        # Chem.MolToMolFile(self.seed, 'seed_before.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_before.mol')
        
        # Align the neighbor ---> attach point vector to the x axis: (0,0,0) ---> (1,0,0)
        # Then translate such that the neighbor is (0,0,0)
        transformations = center_fragment(self.seed,
                                          attach_to_neighbor=False,
                                          neighbor_is_zero=True)
        
        rotation, translation = transformations
        rotate_conformer(conformer=self.pocket_mol.GetConformer(),
                         rotation=rotation)
        translate_conformer(conformer=self.pocket_mol.GetConformer(),
                            translation=translation)
        
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
        
        rotations = []
        for angle in range(-180, 180, 10):
            rotation = Rotation.from_euler('x', angle, degrees=True)
            rotations.append(rotation)
        
        new_fragments = [Fragment(protected_fragment, protected_fragment.protections) for _ in rotations]
        for frag, rotation in zip(new_fragments, rotations):
            rotate_conformer(frag.GetConformer(), rotation=rotation)

        return new_fragments
        
        
    def translate_seed(self,
                       added_fragment: Fragment):
        # translate the seed that had the neighbor at 0 to the neighbor at -interatomic_distance
        # Such that the fragment neighbor that is also the seed attach point is (0,0,0)
        distance = self.get_seed_fragment_distance(added_fragment) # hard coded but should be modified
        
        translation = np.array([-distance, 0, 0])
        translate_conformer(self.seed.GetConformer(), translation=translation)
        translate_conformer(self.pocket_mol.GetConformer(), translation=translation)
        self.transformations.append(translation)
        
        # Chem.MolToMolFile(self.seed, 'seed_after.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_after.mol')
        # import pdb;pdb.set_trace()
        
        
    def action_to_fragment_build(self,
                                 frag_action: int) -> None:
        
        new_fragments = self.get_new_fragments(frag_action)
        new_fragment = new_fragments[0]
        self.translate_seed(added_fragment=new_fragment)
        
        products = [add_fragment_to_seed(seed=self.seed,
                                        fragment=new_fragment)
                    for new_fragment in new_fragments]
        product_clash = [self.get_pocket_ligand_clash(product) for product in products]
        
        default_product = products[0]
        products = [product for product, clash in zip(products, product_clash) if not clash]
        
        if len(products) > 0:
        
            vina_cli = VinaCLI()
            receptor_paths = [self.complex.vina_protein.pdbqt_filepath
                            for _ in products]
            native_ligands = [self.complex.ligand for _ in products]
            frags = [Fragment(product, product.protections) for product in products]
            mols = [self.get_clean_mol(frag) for frag in frags]
            scores = vina_cli.get(receptor_paths=receptor_paths,
                                native_ligands=native_ligands,
                                ligands=mols)
            relative_scores = [score - self.current_score for score in scores]
            rewards = [-score for score in relative_scores]
        
            # product_clash = [self.get_pocket_ligand_clash(product) for product in products]
            # clash_penalty = [0 if not clash else -10 for clash in product_clash]
                
            # penalized_rewards = [reward + penalty for reward, penalty in zip(rewards, clash_penalty)]
        
            # TODO: Add variability : not follow a greedy policy
            # best_reward_i = np.argmax(penalized_rewards)
            best_reward_i = np.argmax(rewards)
            
            product = products[best_reward_i]
            self.current_score = scores[best_reward_i]
            self.seed = product
            
            # reward = penalized_rewards[best_reward_i]
            reward = rewards[best_reward_i]
            
        else:
            self.seed = default_product
            reward = -10
            
        return reward
        
    
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
        
    
    def featurize_pocket(self) -> Data:
        center_pos = [0, 0, 0]
        # seed_copy = Fragment(self.seed, self.seed.protections)
        # seed_copy.protect()
        # seed_atoms = self.rdkit_to_ase(seed_copy)
        # pocket_atoms = self.rdkit_to_ase(self.pocket_mol)
        # if not self.embed_hydrogens:
        #     seed_atoms = seed_atoms[[atom.index for atom in seed_atoms if atom.symbol != 'H']]
        #     pocket_atoms = pocket_atoms[[atom.index for atom in pocket_atoms if atom.symbol != 'H']]
        
        # total_atoms = seed_atoms + pocket_atoms
        # seed_soap = self.soap.create(total_atoms, centers=[center_pos])

        # seed_soap = normalize(seed_soap)

        # return seed_soap.squeeze()
        
        ligand_x, ligand_pos, ligand_focal = self.featurizer.get_fragment_features(fragment=self.seed, 
                                                                    embed_hydrogens=self.embed_hydrogens,
                                                                    center_pos=center_pos)
        protein_x, protein_pos, protein_focal = self.featurizer.get_mol_features(mol=self.pocket_mol,
                                                                embed_hydrogens=self.embed_hydrogens,
                                                                center_pos=center_pos)
        
        pocket_x = protein_x + ligand_x
        pocket_pos = protein_pos + ligand_pos
        is_focal = protein_focal + ligand_focal
        
        mol_id = [0] * len(protein_x) + [1] * len(ligand_x)
        
        # x = torch.tensor(pocket_x, dtype=torch.float)
        x = torch.tensor(pocket_x, dtype=torch.long)
        pos = torch.tensor(pocket_pos, dtype=torch.float)
        mol_id = torch.tensor(mol_id, dtype=torch.long)
        is_focal = torch.tensor(is_focal, dtype=torch.bool)
        
        data = Data(x=x,
                    pos=pos,
                    mol_id=mol_id,
                    is_focal=is_focal,
                    )

        return data
    
    
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
        
        reward = self.action_to_fragment_build(frag_action) # seed is deprotected
        
        if n_actions == self.max_episode_steps: # not terminated but reaching max step size
            self.truncated = True
            # We replace the focal atom with hydrogen (= protect)
            # all other attachment points are already protected
            self.seed.protect() 
        elif not self.terminated:
            self.valid_action_mask = torch.tensor(self.get_valid_action_mask())
            
            # # we terminate the generation if there is no valid action (due to clash)
            # has_valid_action = torch.any(self.valid_action_mask)
            # self.terminated = not has_valid_action
            # if self.terminated:
            #     logging.info('We have an environment with only clashing fragments')
            #     self.seed.protect()
            #     reward = -100 # it is actually a clash penalty
        
        if self.terminated or self.truncated:
            assert(all([atom.GetAtomicNum() > 0 for atom in self.seed.GetAtoms()]))
        
        # observation = self._get_obs()
        info = self._get_info()
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        
        # return observation, reward, self.terminated, self.truncated, info
        return reward, self.terminated, self.truncated, info
    
    
    def set_seed_coordinates(self,
                             pose_mol):
        pose_conformer = pose_mol.GetConformer()
        for transformation in self.transformations:
            if isinstance(transformation, Rotation):
                rotate_conformer(pose_conformer, transformation)
            else:
                translate_conformer(pose_conformer, transformation)
        pose_coordinates = pose_conformer.GetPositions()
        seed_conformer = self.seed.GetConformer()
        for i, new_pos in enumerate(pose_coordinates):
            point = Point3D(*new_pos)
            seed_conformer.SetAtomPosition(i, point)
    
    
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
    
    
    def get_pocket_ligand_clash(self,
                              ligand: Mol):
        
        vdw_distances = defaultdict(dict)
        def get_vdw_min_distance(symbol1, symbol2):
            vdw1 = self.geometry_extractor.get_vdw_radius(symbol1)
            vdw2 = self.geometry_extractor.get_vdw_radius(symbol2)
            
            if symbol1 == 'H':
                min_distance = vdw2
            elif symbol2 == 'H':
                min_distance = vdw1
            else:
                # min_distance = vdw1 + vdw2 - self.clash_tolerance
                min_distance = vdw1 + vdw2
                
            min_distance = min_distance * 0.75
            
            return min_distance
        
        #### Pocket - Seed clashes
        
        ps = Chem.CombineMols(self.pocket_mol, ligand)
        atoms = [atom for atom in ps.GetAtoms()]
        n_pocket_atoms = self.pocket_mol.GetNumAtoms()
        pocket_atoms = atoms[:n_pocket_atoms]
        ligand_atoms = atoms[n_pocket_atoms:]
        distance_matrix = Chem.Get3DDistanceMatrix(mol=ps)
        ps_distance_matrix = distance_matrix[:n_pocket_atoms, n_pocket_atoms:]
        
        # Chem.MolToMolFile(ps, 'pocket_and_seed.mol')
        # import pdb;pdb.set_trace()
        
        # TOO SLOW
        # pocket_pos = self.pocket_mol.GetConformer().GetPositions()
        # pocket_atoms = self.pocket_mol.GetAtoms()
        # fragment_pos = fragment.GetConformer().GetPositions()
        # fragment_atoms = fragment.GetAtoms()
        # distance_matrix = cdist(pocket_pos, fragment_pos)
        # from sklearn import metrics
        # distance_matrix = metrics.pairwise_distances(pocket_pos, fragment_pos)
        
        # import pdb;pdb.set_trace()
        
        has_clash = False
    
        for idx1, atom1 in enumerate(pocket_atoms):
            for idx2, atom2 in enumerate(ligand_atoms):
                
                symbol1 = atom1.GetSymbol()
                symbol2 = atom2.GetSymbol()
                if symbol2 == 'R':
                    symbol2 = 'H' # we mimic that the fragment will bind to a Carbon
                    
                if symbol1 > symbol2:
                    symbol1, symbol2 = symbol2, symbol1
                
                if symbol1 in vdw_distances:
                    if symbol2 in vdw_distances[symbol1]:
                        min_distance = vdw_distances[symbol1][symbol2]
                    else:
                        min_distance = get_vdw_min_distance(symbol1, symbol2)
                        vdw_distances[symbol1][symbol2] = min_distance
                else:
                    min_distance = get_vdw_min_distance(symbol1, symbol2)
                    vdw_distances[symbol1][symbol2] = min_distance
                    
                distance = ps_distance_matrix[idx1, idx2]
                if distance < min_distance:
                    has_clash = True
                    break
                
            if has_clash:
                break
                    
        # Seed - Seed clashes
                    
        if not has_clash:
            
            ss_distance_matrix = distance_matrix[n_pocket_atoms:, n_pocket_atoms:]
            
            # import pdb;pdb.set_trace()

            bonds = self.geometry_extractor.get_bonds(ligand)
            bond_idxs = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        for bond in bonds]
            bond_idxs = [t 
                        if t[0] < t[1] 
                        else (t[1], t[0])
                        for t in bond_idxs ]
            angle_idxs = self.geometry_extractor.get_angles_atom_ids(ligand)
            two_hop_idxs = [(t[0], t[2])
                                    if t[0] < t[2]
                                    else (t[2], t[0])
                                    for t in angle_idxs]
            torsion_idxs = self.geometry_extractor.get_torsions_atom_ids(ligand)
            three_hop_idxs = [(t[0], t[3])
                                if t[0] < t[3]
                                else (t[3], t[0])
                                for t in torsion_idxs]

            for idx1, atom1 in enumerate(ligand_atoms):
                for idx2, atom2 in enumerate(ligand_atoms[idx1+1:]):
                    idx2 = idx2 + idx1 + 1
                    not_bond = (idx1, idx2) not in bond_idxs
                    not_angle = (idx1, idx2) not in two_hop_idxs
                    not_torsion = (idx1, idx2) not in three_hop_idxs
                    if not_bond and not_angle and not_torsion:
                        symbol1 = atom1.GetSymbol()
                        symbol2 = atom2.GetSymbol()
                        if symbol1 == 'R':
                            symbol1 = 'H'
                        if symbol2 == 'R':
                            symbol2 = 'H'
                            
                        if symbol1 > symbol2:
                            symbol1, symbol2 = symbol2, symbol1
                        
                        if symbol1 in vdw_distances:
                            if symbol2 in vdw_distances[symbol1]:
                                min_distance = vdw_distances[symbol1][symbol2]
                            else:
                                min_distance = get_vdw_min_distance(symbol1, symbol2)
                                vdw_distances[symbol1][symbol2] = min_distance
                        else:
                            min_distance = get_vdw_min_distance(symbol1, symbol2)
                            vdw_distances[symbol1][symbol2] = min_distance
                            
                        distance = ss_distance_matrix[idx1, idx2]
                        if distance < min_distance:
                            has_clash = True
                            break
                    
                if has_clash:
                    break
                    
        return has_clash
    
    def get_clean_mol(self,
                      ligand: Mol):
        frag = Fragment(ligand, protections=ligand.protections)
        frag.protect()
        Chem.SanitizeMol(frag)
        mol = Chem.RemoveHs(frag)
        mol_h = Chem.AddHs(mol, addCoords=True)
        # mol_h = Mol(frag)
        
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
    
    
    def get_clean_pocket(self):
        
        new_mol = Mol(self.pocket_mol)
        for transformation in reversed(self.transformations):
            if isinstance(transformation, Rotation):
                rotation_inv = transformation.inv()
                rotate_conformer(new_mol.GetConformer(), rotation=rotation_inv)
            elif isinstance(transformation, np.ndarray):
                translation_inv = -transformation
                translate_conformer(new_mol.GetConformer(), translation=translation_inv)
            else:
                import pdb;pdb.set_trace()
                
        return new_mol
    
    
    def get_minimized_mol(self):
        mol = self.get_clean_mol(self.seed)
        mini_mol = self.complex_minimizer.minimize_ligand(mol, 
                                                        #   distance_constraint=1
                                                          )
        return mini_mol
    
    
class BatchEnv():
    
    def __init__(self,
                 envs: list[FragmentBuilderEnv],
                 memory: dict) -> None:
        self.envs = envs
        self.memory = memory
        self.vina_cli = VinaCLI()
        self.reader = PDBQTReader()
        
        
    def reset(self,
              complexes: list[Complex],
              seeds: list[Fragment],
              initial_scores: list[float],
              absolute_scores: list[float] = None) -> tuple[list[Data], list[dict[str, Any]]]:
        # batch_obs: list[Data] = []
        assert len(complexes) == len(self.envs)
        batch_info: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            info = env.reset(complexes[i],
                             seeds[i],
                             initial_score=initial_scores[i],
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
            valid_action_mask = env.get_valid_action_mask()
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
    
    
    def step(self,
             frag_actions: torch.Tensor,
             ) -> tuple[list[float], list[bool], list[bool], list[dict]]:
        
        assert frag_actions.size()[0] == len(self.ongoing_env_idxs)

        batch_truncated = []
        batch_info = []
        batch_rewards = []
        for env_i, frag_action in zip(self.ongoing_env_idxs, frag_actions):
            env = self.envs[env_i]
            frag_action = int(frag_action)
            reward, terminated, truncated, info = env.step(frag_action)
            
            if terminated: # we have no attachment points left
                self.terminateds[env_i] = True
            
            batch_truncated.append(truncated)
            batch_info.append(info)
            batch_rewards.append(reward)
        
        # Truncation means that we have reached the maximum number of step
        # The molecule is likely too big, so we penalize
        for i, (reward, truncated) in enumerate(zip(batch_rewards, batch_truncated)):
            if truncated:
                batch_rewards[i] = -10.0
        
        # this will cause truncation to extend to termination
        for env_i, reward in zip(self.ongoing_env_idxs, batch_rewards):
            if reward < 0: # there is a clash, or bad interaction, or too many atoms
                self.terminateds[env_i] = True
                
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            terminated = self.terminateds[env_i]
            if terminated:
                env.seed.protect()
            else:
                env.seed_to_frame()
            
        # Refresh the ongoing envs, that are the one not terminated
        self.ongoing_env_idxs = [i 
                                 for i, terminated in enumerate(self.terminateds) 
                                 if not terminated]
        logging.debug(self.ongoing_env_idxs)
        logging.debug(self.terminateds)
        
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