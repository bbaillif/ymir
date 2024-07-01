import numpy as np
import torch
import logging
import io
import os
import copy
import time

from rdkit import Chem
from rdkit.Chem import Mol
from ase.io import read
from ase import Atoms
from ymir.molecule_builder import add_fragment_to_seed, check_assembly
from ymir.utils.fragment import (get_fragments_from_mol, 
                                 get_neighbor_symbol, 
                                 center_fragment,
                                 find_mappings, 
                                 get_neighbor_id_for_atom_id)
from ymir.data.structure.complex import Complex
from torch_geometric.data import Data, Batch
from ymir.data import Fragment
from typing import Any
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean
from ymir.utils.spatial import rotate_conformer, translate_conformer, rdkit_distance_matrix
from ymir.geometry.geometry_extractor import GeometryExtractor
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist
from ymir.atomic_num_table import AtomicNumberTable
from ymir.params import EMBED_HYDROGENS, SCORING_FUNCTION, TORSION_ANGLES_DEG
from ymir.featurizer_void import Featurizer
from ymir.metrics.activity import VinaScore
from ymir.bond_distance import MedianBondDistance
from ymir.metrics.activity.vina_cli import VinaCLI
from ymir.metrics.activity.smina_cli import SminaCLI
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate
from rdkit.Geometry import Point3D
from ymir.pdbqt_reader import PDBQTReader
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from scipy.spatial.distance import euclidean
from ymir.complex_minimizer import ComplexMinimizer
from ymir.save_state import StateSave, Memory
from ymir.data.structure import GlideProtein
from ymir.metrics.activity import GlideScore
from rdkit.Chem.rdMolAlign import CalcRMS
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
import gzip
import shutil
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
from ymir.utils.spatial import add_noise
from ymir.molecule_builder import potential_reactions


class FragmentBuilderEnv():
    
    def __init__(self,
                 rotated_fragments: list[list[Fragment]],
                 attach_labels: list[list[int]],
                 z_table: AtomicNumberTable,
                 max_episode_steps: int = 10,
                 valid_action_masks: dict[int, torch.Tensor] = None,
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 pocket_feature_type: str = 'soap',
                 ) -> None:
        self.rotated_fragments = rotated_fragments
        self.attach_labels = attach_labels
        self.z_table = z_table
        self.max_episode_steps = max_episode_steps
        self.valid_action_masks = valid_action_masks
        self.embed_hydrogens = embed_hydrogens
        self.pocket_feature_type = pocket_feature_type
        
        self.n_fragments = len(self.rotated_fragments)
        self._action_dim = self.n_fragments
        
        for mask in self.valid_action_masks.values():
            assert mask.size()[-1] == self.n_fragments

        self.seed: Fragment = None
        self.fragment: Fragment = None
        
        self.geometry_extractor = GeometryExtractor()
        self.featurizer = Featurizer(z_table=self.z_table)
        self.mbd = MedianBondDistance()
        
        species = ['C', 'N', 'O', 'Cl', 'S']
        # if self.embed_hydrogens:
        #     species = ['H'] + species
        self.soap = SOAP(species=species,
                        r_cut=8,
                        n_max=8,
                        l_max=6)
        
        
    @property
    def action_dim(self) -> int:
        return self._action_dim
        
        
    def reset(self, 
              complx: Complex,
              seed: Fragment,
              is_real: bool,
              generation_path: str,
              initial_score: float,
              native_score: float,
            #   memory: dict
              ) -> tuple[Data, dict[str, Any]]:
        
        self.complex = complx
        self.original_seed = Fragment.from_fragment(seed)
        self.seed = Fragment.from_fragment(seed)
        self.is_real = is_real
        self.generation_path = generation_path
        # self.memory = memory
        self.initial_score = initial_score
        self.current_score = initial_score
        self.native_score = native_score
        
        # self.pocket_mol = Chem.RemoveHs(self.complex.pocket.mol)
        self.pocket_mol = Chem.MolFromPDBFile(self.complex.pocket_path, 
                                              sanitize=False)
        
        self.actions = []
        
        self.original_seed = Fragment.from_fragment(self.seed)
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        assert self.terminated is False, 'No attachement point in the seed'
        self.truncated = False
        
        assert not self.terminated
        
        # self.seed_to_frame()
        
        # observation = self._get_obs()
        info = self._get_info()
        
        has_valid_action = torch.any(self.get_valid_action_mask())
        self.terminated = not has_valid_action
        
        # this should not happen, unless the set of fragments is 
        # not suitable for the given starting situation
        if self.terminated: 
            import pdb;pdb.set_trace()
        
        # self.complex_minimizer = ComplexMinimizer(pocket=self.complex.pocket)
        
        self.new_atom_idxs = range(self.seed.mol.GetNumAtoms())
        
        
        
        # return observation, info
        return info
        
        
    def get_seed_fragment_distance(self,
                                   fragment: Fragment):
        # Get fragment attach neighbor (which is the seed attach point)
        # Get Seed attach neighbor (which is the fragment attach point)
        # Compute the ideal length between the two atoms
        
        construct_neighbor_symbol = get_neighbor_symbol(self.seed)
        fragment_neighbor_symbol = get_neighbor_symbol(fragment)
        distance = self.mbd.get_mbd(construct_neighbor_symbol, fragment_neighbor_symbol)
        return distance
        
        
    def get_new_fragments(self,
                        frag_action: int) -> list[Fragment]:
        
        rotated_fragments = copy.deepcopy(self.rotated_fragments[frag_action])

        return rotated_fragments
        
        
    def fragments_to_frame(self,
                           fragments: list[Fragment]):
        
        ideal_distance = self.get_seed_fragment_distance(fragments[0])
        # Set focal distance
        neighbor_id = get_neighbor_id_for_atom_id(mol=self.seed.mol,
                                                atom_id=self.focal_atom_id)
        seed_conf = self.seed.mol.GetConformer()
        seed_positions = seed_conf.GetPositions()
        neighbor_position = seed_positions[neighbor_id]
        focal_position = seed_positions[self.focal_atom_id]
        direction = focal_position - neighbor_position
        current_distance = np.linalg.norm(direction)
        scale = ideal_distance / current_distance
        focal_translation = direction * (scale - 1)
        new_focal_position = focal_position + focal_translation
        point3D = Point3D(*new_focal_position)
        seed_conf.SetAtomPosition(self.focal_atom_id, point3D)
        neigh_to_focal = new_focal_position - neighbor_position
        
        # neighbor_centred_seed = Fragment.from_fragment(self.seed)
        # translate_conformer(neighbor_centred_seed.mol.GetConformer(), translation=-neighbor_position)
        
        new_frag_neigh_position = np.array([ideal_distance, 0, 0]) # cause current neigh position is [0,0,0]
        rotation, rssd = Rotation.align_vectors(a=neigh_to_focal.reshape(-1, 3), b=new_frag_neigh_position.reshape(-1, 3))
        for new_fragment in fragments:
            attach_points = new_fragment.get_attach_points()
            attach_id = list(attach_points.keys())[0]
            frag_neighbor_id = get_neighbor_id_for_atom_id(mol=new_fragment.mol, 
                                                           atom_id=attach_id)
            
            # Set fragment attach to [0,0,0] and neighbor to [ideal_distance, 0, 0]
            translate_conformer(new_fragment.mol.GetConformer(), translation=new_frag_neigh_position)
            frag_conf = new_fragment.mol.GetConformer()
            frag_conf.SetAtomPosition(attach_id, Point3D(0, 0, 0))
            
            # Rotate fragment to align (frag_attach -> frag_neigh) to (seed_neigh -> seed_focal)
            rotate_conformer(frag_conf, rotation=rotation)
            
            # Translate fragment to superpose seed_focal to frag_neigh
            translate_conformer(frag_conf, translation=neighbor_position)
            try:
                assert np.allclose(frag_conf.GetPositions()[attach_id], neighbor_position)
                assert np.allclose(frag_conf.GetPositions()[frag_neighbor_id], new_focal_position)
            except:
                import pdb;pdb.set_trace()
        
        
    def action_to_products(self,
                            frag_action: int) -> tuple[list[Fragment], list[list[int]]]:
        
        new_fragments = self.get_new_fragments(frag_action)
        
        # Set attach label to a label among the possible for the chosen fragment
        potential_labels = self.attach_labels[frag_action]
        seed_label = self.attach_points[self.focal_atom_id]
        potential_reactions_seed = potential_reactions[seed_label]
        for potential_label in potential_labels:
            if potential_label in potential_reactions_seed:
                # attach_atom = self.seed.mol.GetAtomWithIdx(self.focal_atom_id)
                # attach_atom.SetIsotope(potential_label)
                for new_fragment in new_fragments:
                    attach_atom = new_fragment.mol.GetAtomWithIdx(list(new_fragment.get_attach_points().keys())[0])
                    attach_atom.SetIsotope(potential_label)
                break
        
        # seed_mol = Chem.RemoveHs(seed_mol)
        self.fragments_to_frame(new_fragments)
        
        tpls = [add_fragment_to_seed(seed=self.seed,
                                        fragment=new_fragment)
                    for new_fragment in new_fragments]
        products = [tpl[0] for tpl in tpls]
        f2p_mappings = [tpl[2] for tpl in tpls]
        all_new_atom_idxs: list[list[int]] = []
        for f2p_mapping, fragment in zip(f2p_mappings, new_fragments):
            aps = fragment.get_attach_points()
            assert len(aps) == 1
            attach_point = list(aps.keys())[0]
            f2p_mapping.pop(attach_point)
            # new_atom_idxs = list(f2p_mapping.values())
            all_new_atom_idxs.append(f2p_mapping)
        
        # products = [add_fragment_to_seed(seed=self.seed,
        #                                 fragment=new_fragment)[0]
        #             for new_fragment in new_fragments]
        # product_clash = [self.get_pocket_ligand_clash(product.mol) for product in products]
        
        # default_product = products[0]
        # products = [product for product, clash in zip(products, product_clash) if not clash]
        
        # if len(products) == 0:
        #     products = [default_product]
        
        return products, all_new_atom_idxs
        
    
    def _get_obs(self):
        data = self.featurize_pocket()
        obs = data
        return obs
    
    
    def _get_info(self):
        return {}
    
    
    def rdkit_to_ase(self,
                     mol: Mol):
        # filename = 'mol.xyz'
        # Chem.MolToXYZFile(mol, filename)
        # molblock = Chem.MolToXYZBlock(mol)
        # f = io.StringIO(molblock)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = mol.GetConformer().GetPositions()
        # ase_atoms = read(f)
        ase_atoms = Atoms(symbols=symbols, positions=positions)
        return ase_atoms
        
    
    def featurize_pocket(self) -> Data:
        center_pos = self.seed.mol.GetConformer().GetAtomPosition(self.focal_atom_id)
        center_pos = np.array([center_pos.x, center_pos.y, center_pos.z])
        if self.pocket_feature_type == 'soap':
            seed_copy = Fragment.from_fragment(self.seed)
            seed_copy.protect()
            pocket_copy = Mol(self.pocket_mol)
            
            add_noise(seed_copy.mol.GetConformer())
            add_noise(pocket_copy.GetConformer())
            
            distance_matrix = rdkit_distance_matrix(seed_copy.mol, pocket_copy)
            distance_to_center = distance_matrix[:, self.focal_atom_id]
            seed_dist_to_center = distance_to_center[:seed_copy.mol.GetNumAtoms()]
            pocket_dist_to_center = distance_to_center[seed_copy.mol.GetNumAtoms():]
            
            assert len(seed_dist_to_center) == seed_copy.mol.GetNumAtoms()
            assert len(pocket_dist_to_center) == pocket_copy.GetNumAtoms()
            
            seed_atoms = self.rdkit_to_ase(seed_copy.mol)
            pocket_atoms = self.rdkit_to_ase(pocket_copy)
            # if not self.embed_hydrogens:
            #     try:
            selected_seed_atom_idxs = [atom.index
                                    for atom, distance in zip(seed_atoms, seed_dist_to_center) 
                                    if (atom.symbol != 'H') and (distance < 10)]
            seed_atoms = seed_atoms[selected_seed_atom_idxs]
            selected_pocket_atom_idxs = [atom.index 
                                        for atom, distance in zip(pocket_atoms, pocket_dist_to_center) 
                                        if (atom.symbol != 'H') and (distance < 10)]
            if len(selected_pocket_atom_idxs) > 0:
                pocket_atoms = pocket_atoms[selected_pocket_atom_idxs]
                total_atoms = seed_atoms + pocket_atoms
            else:
                total_atoms = seed_atoms
                # except:
                #     import pdb;pdb.set_trace()
            
            seed_soap = self.soap.create(total_atoms, centers=[center_pos])

            seed_soap = normalize(seed_soap)

            return seed_soap.squeeze()
        
        else:
        
            dummy_points = []
            padding = 6
            resolution = 2
            for x1 in np.arange(center_pos[0] - padding, center_pos[0] + padding + 1, resolution):
                for y1 in np.arange(center_pos[1] - padding, center_pos[1] + padding + 1, resolution):
                    for z1 in np.arange(center_pos[2] - padding, center_pos[2] + padding + 1, resolution):
                        dummy_points.append([x1, y1, z1])
                        
            dummy_points = np.array(dummy_points)
        
            if not self.embed_hydrogens:
                pocket_mol = Chem.RemoveHs(self.pocket_mol, sanitize=False)
            pocket_lig_mol = Chem.CombineMols(pocket_mol, self.seed.mol)
            pocket_pos = pocket_lig_mol.GetConformer().GetPositions()
            
            # not_focal = [True if i != (pocket_mol.GetNumAtoms() + self.focal_atom_id) else False for i in range(pocket_lig_mol.GetNumAtoms())]
            # pocket_pos = pocket_pos[not_focal]
            
            distance_matrix = cdist(dummy_points, pocket_pos)
            distance_to_focal = distance_matrix[:, pocket_mol.GetNumAtoms() + self.focal_atom_id]
            neigh_id = get_neighbor_id_for_atom_id(mol=self.seed.mol, atom_id=self.focal_atom_id)
            distance_to_neigh = distance_matrix[:, pocket_mol.GetNumAtoms() + neigh_id]
            
            distance_matrix_nonfocal = np.delete(distance_matrix, pocket_mol.GetNumAtoms() + self.focal_atom_id, axis=1)
            min_distance = np.min(distance_matrix_nonfocal, axis=1)
            
            dummy_points = dummy_points[(min_distance > 2) 
                                        & (distance_to_focal < padding) 
                                        & (distance_to_focal < distance_to_neigh)]
            dummy_x = [self.z_table.z_to_index(0)] * len(dummy_points)
            dummy_focal = [0] * len(dummy_points)
            dummy_pos = dummy_points.tolist()
        
            ligand_x, ligand_pos, ligand_focal = self.featurizer.get_fragment_features(fragment=self.seed, 
                                                                        embed_hydrogens=self.embed_hydrogens,
                                                                        center_pos=center_pos,
                                                                        dummy_points=dummy_points)
            protein_x, protein_pos, protein_focal = self.featurizer.get_mol_features(mol=self.pocket_mol,
                                                                    embed_hydrogens=self.embed_hydrogens,
                                                                    center_pos=center_pos,
                                                                    dummy_points=dummy_points)
            
            pocket_x = dummy_x + protein_x + ligand_x
            pocket_pos = dummy_pos + protein_pos + ligand_pos
            is_focal = dummy_focal + protein_focal + ligand_focal
            
            mol_id = [0] * len(dummy_x) + [1] * len(protein_x) + [2] * len(ligand_x)
            
            # with open('dummy.xyz', 'w') as f:
            #     n_dummies = len(dummy_pos)
            #     f.write(str(n_dummies) + '\n')
            #     f.write('\n')
            #     for pos in dummy_pos:
            #         f.write(f'H {pos[0]} {pos[1]} {pos[2]}\n')
            
            # import pdb;pdb.set_trace()
            
            # x = torch.tensor(pocket_x, dtype=torch.float)
            x = torch.tensor(pocket_x, dtype=torch.long)
            pos = torch.tensor(pocket_pos, dtype=torch.float)
            mol_id = torch.tensor(mol_id, dtype=torch.long)
            # is_focal = torch.tensor(is_focal, dtype=torch.bool)
            is_focal = torch.tensor(is_focal, dtype=torch.long)
            
            noise = torch.randn_like(pos) * 0.01
            pos = pos + noise
            
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
            # if self.is_real:
                
            #     n_actions = len(self.actions)
            #     focal_ideal_position = self.generation_path[n_actions]
            #     attach_points = list(self.seed.get_attach_points().keys())
            #     seed_positions = self.seed.mol.GetConformer().GetPositions()
            #     distances = [euclidean(seed_positions[attach_point], focal_ideal_position)
            #                 for attach_point in attach_points]
            #     self.focal_atom_id = attach_points[np.argmin(distances)]
                
            # else:
            # # self.focal_atom_id = np.random.choice(list(self.attach_points.keys()))
            # # to ensure it is deterministic to work with the memory of the BatchEnv
            #     self.focal_atom_id = list(self.attach_points.keys())[0]
                
            self.focal_atom_id = list(self.attach_points.keys())[0]
                
            self.seed.protect(atom_ids_to_keep=[self.focal_atom_id])
            assert (len(self.seed.get_attach_points()) == 1)
        return terminated
    
    
    def step(self,
             frag_action: int,
             product: Fragment,
             score: float = 0,
             reward: float = 0,):
        
        self.actions.append(frag_action)
        self.seed = Fragment.from_fragment(product)
        self.current_score = score
        
        if np.isnan(self.current_score):
            self.current_score = 1
        
        # observation = self._get_obs()
        info = self._get_info()
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        
        if reward <= 0:
            self.terminated = True
        self.last_reward = reward
        
        n_actions = len(self.actions)
        if (n_actions == self.max_episode_steps) and (not self.terminated): # not terminated but reaching max step size
            self.truncated = True
            # We replace the focal atom with hydrogen
            # all other attachment points are already protected
            self.seed.protect(protection_atomic_num=1) 
        
        # if self.terminated or self.truncated:
        #     assert(all([atom.GetAtomicNum() > 0 for atom in self.seed.mol.GetAtoms()]))
        
        # return observation, reward, self.terminated, self.truncated, info
        return self.terminated, self.truncated, info
    
    
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
    
    def get_clean_fragment(self,
                            fragment: Fragment,
                            protection_atomic_num: int = 1,
                            add_hydrogens: bool = False) -> Mol:
        clean_frag = Fragment.from_fragment(fragment)
        # protect with C to force the placement of attach points to empty areas
        clean_frag.protect(protection_atomic_num=protection_atomic_num) 
        mol = clean_frag.mol
        # Chem.SanitizeMol(mol)
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)
            # mol = Chem.AddHs(mol, addCoords=True)
        # mol_h = Mol(frag)
        
        return mol
    
    
    def get_clean_pocket(self):
        
        new_mol = Mol(self.pocket_mol)
        return new_mol
    
    
    def get_minimized_mol(self):
        mol = self.get_clean_fragment(self.seed)
        mini_mol = self.complex_minimizer.minimize_ligand(mol, 
                                                        #   distance_constraint=1
                                                          )
        return mini_mol
    
    
    
class BatchEnv():
    
    def __init__(self,
                 envs: list[FragmentBuilderEnv],
                 memory: Memory,
                 best_scores: dict[int, float],
                 pocket_feature_type: str = 'soap',
                 scoring_function: str = SCORING_FUNCTION,
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 ) -> None:
        self.envs = envs
        self.pocket_feature_type = pocket_feature_type
        self.memory = memory
        self.best_scores = best_scores
        self.embed_hydrogens = embed_hydrogens
        self.vina_cli = VinaCLI()
        self.reader = PDBQTReader()
        
        assert scoring_function in ['vina', 'glide', 'smina']
        self.scoring_function = scoring_function
        
        self.species = ['C', 'N', 'O', 'Cl', 'S']
        # if self.embed_hydrogens:
        #     species = ['H'] + species
        self.soap = SOAP(species=self.species,
                        r_cut=8,
                        n_max=8,
                        l_max=6)
        
        self.pocket_feature_dim = self.soap.get_number_of_features()
        
        
    def reset(self,
              complexes: list[Complex],
              seeds: list[Fragment],
              initial_scores: list[float],
              native_scores: list[float],
              seed_idxs: list[int],
              real_data_idxs: list[int],
              current_generation_paths: list[tuple[float, str]],
              absolute_scores: list[float] = None) -> tuple[list[Data], list[dict[str, Any]]]:
        # batch_obs: list[Data] = []
        assert len(complexes) == len(self.envs)
        batch_info: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            is_real = i in real_data_idxs
            info = env.reset(complexes[i],
                             seeds[i],
                             is_real,
                             current_generation_paths[i],
                             initial_score=initial_scores[i],
                             native_score=native_scores[i],
                             
                            #  self.memory
                             )
            batch_info.append(info)
        self.terminateds = [False] * len(self.envs)
        self.truncateds = [False] * len(self.envs)
        self.ongoing_env_idxs = list(range(len(self.envs)))
        
        self.current_scores = initial_scores
        self.absolute_scores = absolute_scores
        self.native_scores = native_scores
        
        self.seed_idxs = seed_idxs
        
        return batch_info
    
    
    def get_valid_action_mask(self) -> torch.Tensor:
        masks = []
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            valid_action_mask = env.get_valid_action_mask()
            masks.append(valid_action_mask)
        masks = torch.stack(masks)
        # non_terminated = [not terminated for terminated in self.terminateds]
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
        
        frag_actions: list[int] = [int(frag_action) for frag_action in frag_actions]
        assert len(frag_actions) == len(self.ongoing_env_idxs)

        # Determine action to score
        batch_truncated = []
        tested_products: list[Fragment] = []
        tested_new_atom_idxs = []
        tested_env_idxs: list[int] = []
        tested_states: dict[tuple[int], list[int]] = {}
        state_saves: dict[tuple[float], StateSave] = {}
        # start_time = time.time()
        for env_i, frag_action in zip(self.ongoing_env_idxs, frag_actions):
            env = self.envs[env_i]
            
            seed_idx = self.seed_idxs[env_i]
            state = (seed_idx, *env.actions, frag_action)
            # if state in self.memory:
            #     state_save = self.memory[state]
            #     state_saves[env_i] = state_save
            # elif state in tested_states:
            if state in tested_states:
                tested_states[state].append(env_i)
            # if (not state in self.memory) and (state not in tested_states):
            else:
                # try:
                products, new_atom_idxs = env.action_to_products(frag_action)
                # except Exception as exception:
                #     print(str(exception))
                #     import pdb;pdb.set_trace()
                #     print(str(exception))
                # products are unprotected
                for product in products:
                    # Add carbon for mininplace
                    product.protect()
                    if self.scoring_function == 'glide':
                        mol_h = Chem.AddHs(product.mol, addCoords=True)
                        if (product.mol.GetNumAtoms() + 3 * len(product.protections)) != mol_h.GetNumAtoms():
                            import pdb;pdb.set_trace()
                        product.mol = mol_h
                    else:
                        product.mol = Chem.RemoveHs(product.mol, updateExplicitCount=True)
                    
                # Maybe sanitize
                tested_products.extend(products)
                tested_new_atom_idxs.extend(new_atom_idxs)
                tested_env_idxs.extend([env_i] * len(products))
                tested_states[state] = [env_i]
            
        # logging.info(f'Time to get products: {time.time() - start_time}')
            
        if len(tested_products) > 0:
            
            if self.scoring_function == 'glide':
            
                # Glide docking
                env = self.envs[0]
                glide_protein = GlideProtein(pdb_filepath=env.complex.vina_protein.protein_clean_filepath,
                                                native_ligand=env.complex.ligand)
                glide = GlideScore(glide_protein, mininplace=True)
                mols = [product.mol for product in tested_products]
                scores = glide.get(mols=mols)
                scores = np.array(scores)
            
                if os.path.exists(glide.pose_filepath):
                    docked_poses = [mol 
                                    for mol in Chem.SDMolSupplier(glide.pose_filepath, removeHs=False)]
                    for pose in docked_poses:
                        if pose is None:
                            import pdb;pdb.set_trace()
                else:
                    docked_poses = []
                    
                is_nan = np.isnan(scores)
                valid_idxs = [i for i, isnan in enumerate(is_nan) if not isnan]
                if len(valid_idxs) != len(docked_poses):
                    import pdb;pdb.set_trace()
                    
                for env_i in set(tested_env_idxs):
                    env = self.envs[env_i]
                    seed_score = env.current_score
                    tested_i_env = [i 
                                    for i, env_i2 in enumerate(tested_env_idxs) 
                                    if env_i == env_i2]
                    valid_tested_i_env = [i 
                                        for i in tested_i_env
                                        if i in valid_idxs]
                    if len(valid_tested_i_env) > 0:
                        rewards = []
                        rmsds = []
                        for test_i in valid_tested_i_env:
                            valid_i = valid_idxs.index(test_i)
                            pose = docked_poses[valid_i]
                            score = scores[test_i]
                            relative_score = score - seed_score
                            product = tested_products[test_i]
                            new_atom_idxs = tested_new_atom_idxs[test_i]
                            
                            try:
                                other_atom_idxs = [i 
                                                for i in range(product.mol.GetNumAtoms()) 
                                                if i not in new_atom_idxs]
                            except:
                                import pdb;pdb.set_trace()
                            no_hydrogen_atom_idxs = [i 
                                                    for i in other_atom_idxs 
                                                    if product.mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
                            try:
                                pose_seed_positions = pose.GetConformer().GetPositions()[no_hydrogen_atom_idxs]
                                product_seed_positions = product.mol.GetConformer().GetPositions()[no_hydrogen_atom_idxs]
                            except:
                                import pdb;pdb.set_trace()
                            deviation = pose_seed_positions - product_seed_positions
                            sd = np.square(deviation)
                            msd = np.mean(sd)
                            rmsd = np.sqrt(msd)
                            if np.isnan(rmsd):
                                import pdb;pdb.set_trace()
                            # rmsd_map = [[atom_idx, atom_idx] for atom_idx in other_atom_idxs]
                            # rmsd = CalcRMS(pose, product.mol, map=rmsd_map)
                            rmsds.append(rmsd)
                            reward = - relative_score - rmsd
                            rewards.append(reward)
                            
                        best_reward_i = np.argmax(rewards)
                        rmsd_best = rmsds[best_reward_i]
                        best_test_i = valid_tested_i_env[best_reward_i]
                        best_valid_i = valid_idxs.index(best_test_i)
                        best_pose = docked_poses[best_valid_i]
                        best_score = scores[best_test_i]
                        new_seed = tested_products[best_test_i]
                        if new_seed.mol.GetNumAtoms() != best_pose.GetNumAtoms():
                            import pdb;pdb.set_trace()
                        new_seed.mol.RemoveAllConformers()
                        new_seed.mol.AddConformer(best_pose.GetConformer())
                        
                    else:
                        # Setup default product
                        new_seed = tested_products[tested_i_env[0]]
                        best_score = np.nan
                        rmsd_best = np.nan
                        
                    new_seed.unprotect()
                    state_save = StateSave(score=best_score,
                                            rmsd=rmsd_best,
                                            seed=new_seed)
                    state_saves[env_i] = state_save
                    
            elif self.scoring_function == 'smina':
                receptor_filepaths = []
                for env_i in tested_env_idxs:
                    env = self.envs[env_i]
                    receptor_filepaths.append(env.complex.vina_protein.pdbqt_filepath)
                smina_cli = SminaCLI(score_only=False)
                ligands = [product.mol for product in tested_products]
                scores = smina_cli.get(receptor_paths=receptor_filepaths,
                                        ligands=ligands)
                docked_poses = smina_cli.get_poses()
                
                for env_i in set(tested_env_idxs):
                    env = self.envs[env_i]
                    seed_score = env.current_score
                    tested_i_env = [i 
                                    for i, env_i2 in enumerate(tested_env_idxs) 
                                    if env_i == env_i2]
                    
                    rewards = []
                    rmsds = []
                    for test_i in tested_i_env:
                        pose = docked_poses[test_i]
                        score = scores[test_i]
                        relative_score = score - seed_score
                        product = tested_products[test_i]
                        
                        pose2product_matches = product.mol.GetSubstructMatches(pose)
                        matches_rmsds = []
                        pose_copies = []
                        if len(pose2product_matches) > 1:
                            Chem.MolToMolFile(pose, 'pose_multiple_match.mol')
                            Chem.MolToMolFile(product.mol, 'product_multiple_match.mol')
                        for pose2product in pose2product_matches:
                            if len(pose2product) != product.mol.GetNumAtoms():
                                import pdb;pdb.set_trace()
                            pose_copy = Mol(pose)
                            for pose_atom_id1, product_atom_id2 in enumerate(pose2product):
                                point3d = pose.GetConformer().GetAtomPosition(pose_atom_id1)
                                try:
                                    pose_copy.GetConformer().SetAtomPosition(product_atom_id2, point3d)
                                except:
                                    import pdb;pdb.set_trace()
                            
                            new_atom_idxs = tested_new_atom_idxs[test_i]
                            other_atom_idxs = [i 
                                            for i in range(product.mol.GetNumAtoms()) 
                                            if i not in new_atom_idxs]
                            pose_seed_positions = pose.GetConformer().GetPositions()[other_atom_idxs]
                            product_seed_positions = product.mol.GetConformer().GetPositions()[other_atom_idxs]
                            deviation = pose_seed_positions - product_seed_positions
                            sd = np.square(deviation)
                            msd = np.mean(sd)
                            rmsd = np.sqrt(msd)
                            if np.isnan(rmsd):
                                import pdb;pdb.set_trace()
                                
                            matches_rmsds.append(rmsd)
                            pose_copies.append(pose_copy)
                            
                        try:
                            best_rmsd_i = np.argmin(matches_rmsds)
                        except:
                            import pdb;pdb.set_trace()
                        pose_copy = pose_copies[best_rmsd_i]
                        rmsd = matches_rmsds[best_rmsd_i]
                        docked_poses[test_i] = pose_copy
                        pose = docked_poses[test_i]
                                
                        rmsds.append(rmsd)
                            
                        reward = - relative_score - rmsd / 2
                        rewards.append(reward)
                        
                    best_reward_i = np.argmax(rewards)
                    rmsd_best = rmsds[best_reward_i]
                    best_test_i = tested_i_env[best_reward_i]
                    best_pose = docked_poses[best_test_i]
                    best_score = scores[best_test_i]
                    new_seed = tested_products[best_test_i]
                    if new_seed.mol.GetNumAtoms() != best_pose.GetNumAtoms():
                        import pdb;pdb.set_trace()
                    new_seed.mol.RemoveAllConformers()
                    new_seed.mol.AddConformer(best_pose.GetConformer())
                
                    new_seed.unprotect()
                    state_save = StateSave(score=best_score,
                                            rmsd=rmsd_best,
                                            seed=new_seed)
                    state_saves[env_i] = state_save
                    # import pdb;pdb.set_trace()
            
            # Complete duplicate states
            for state in tested_states:
                env_idxs = tested_states[state]
                if len(env_idxs) > 1:
                    for env_i in env_idxs:
                        state_saves[env_i] = state_saves[env_idxs[0]]
                    # if not state in self.memory:
                    #     self.memory[state] = state_saves[env_i]
        
        assert len(state_saves) == len(self.ongoing_env_idxs)
        
        # Refresh best scores
        for env_i, frag_action in zip(self.ongoing_env_idxs, frag_actions):
            env = self.envs[env_i]
            seed_idx = self.seed_idxs[env_i]
            state_save = state_saves[env_i]
            best_score = self.best_scores[seed_idx]
            if state_save.score < best_score:
                logging.info(f'New best score: {state_save.score} better than {best_score} for seed {seed_idx}')
                self.best_scores[seed_idx] = state_save.score
        
        # Step with correct products
        rewards = []
        for env_i, frag_action in zip(self.ongoing_env_idxs, frag_actions):
            env = self.envs[env_i]
            state_save = state_saves[env_i]
            seed_idx = self.seed_idxs[env_i]
            state = (seed_idx, *env.actions, frag_action)
            # if not state in self.memory:
            #     self.memory[state] = state_save
                
            best_score = self.best_scores[seed_idx]
            
            if np.isnan(state_save.score):
                reward = 0
            else:
                size_malus = 0
                seed_nha = state_save.seed.mol.GetNumHeavyAtoms()
                if seed_nha > 50:
                    # size_malus = seed_nha - 50
                    reward = 0
                else:
                    seed_score = env.current_score
                    # to_scale = lambda score: (score - env.initial_score) / best_score - env.initial_score
                    # relative_scaled_score = to_scale(state_save.score) - to_scale(seed_score)
                    # reward = max(relative_scaled_score, 0) # - state_save.rmsd
                    relative_score = state_save.score - seed_score
                    reward = max(-relative_score, 0)
            if np.isnan(reward):
                import pdb;pdb.set_trace()
            rewards.append(reward)
            
            terminated, truncated, info = env.step(frag_action, 
                                                    product=state_save.seed,
                                                    score=state_save.score,
                                                    reward=reward)
            
            if truncated:
                terminated = True
                self.truncateds[env_i] = True
            
            if terminated: # we have no attachment points left
                self.terminateds[env_i] = True
            
            batch_truncated.append(truncated)
            
            # if terminated:
            #     env.seed.protect()
            
        # Refresh the ongoing envs, that are the one not terminated
        self.ongoing_env_idxs = [i 
                                 for i, terminated in enumerate(self.terminateds) 
                                 if not terminated]
        logging.debug(self.ongoing_env_idxs)
        logging.debug(self.terminateds)
        
        return rewards, list(self.terminateds), list(self.truncateds)
        
            
    def get_clean_fragments(self) -> list[Mol]:
        
        mols = []
        for env in self.envs:
            mol_h = env.get_clean_fragment(env.seed)
            mols.append(mol_h)
        return mols
            
            
    def get_obs(self,
                all_envs: bool = False) -> list[Data]:
        obs_list = []
        if all_envs:
            env_idxs = range(len(self.envs))
        else: # take ongoing
            env_idxs = self.ongoing_env_idxs
            
        if self.pocket_feature_type == 'soap':
            center_pos_list = []
            all_atoms = []
            for env_i in env_idxs:
                
                env = self.envs[env_i]
                center_pos = env.seed.mol.GetConformer().GetAtomPosition(env.focal_atom_id)
                center_pos = np.array([center_pos.x, center_pos.y, center_pos.z])
                
                seed_copy = Fragment.from_fragment(env.seed)
                seed_copy.protect(protection_atomic_num=1)
                # pocket_copy = Mol(env.pocket_mol)
                
                neigh_id = get_neighbor_id_for_atom_id(mol=seed_copy.mol, atom_id=env.focal_atom_id)
                
                # add_noise(seed_copy.mol.GetConformer())
                # add_noise(pocket_copy.GetConformer())
                # distance_matrix = rdkit_distance_matrix(seed_copy.mol, env.pocket_mol)
                # distance_to_center = distance_matrix[:, env.focal_atom_id]
                
                # distance_to_neigh = distance_matrix[:, neigh_id]
                seed_positions = seed_copy.mol.GetConformer().GetPositions()
                pocket_positions = env.pocket_mol.GetConformer().GetPositions()
                all_positions = np.concatenate([seed_positions, pocket_positions])
                neigh_position = seed_positions[neigh_id]
                distance_matrix = cdist(all_positions, np.array([center_pos, neigh_position]))
                distance_to_center = distance_matrix[:, 0]
                distance_to_neigh = distance_matrix[:, 1]
                
                seed_dist_to_center = distance_to_center[:seed_copy.mol.GetNumAtoms()]
                seed_dist_to_neigh = distance_to_neigh[:seed_copy.mol.GetNumAtoms()]
                assert len(seed_dist_to_center) == seed_copy.mol.GetNumAtoms()
                seed_symbols = [atom.GetSymbol() for atom in seed_copy.mol.GetAtoms()]
                seed_positions = seed_copy.mol.GetConformer().GetPositions()
                seed_positions = seed_positions + np.random.default_rng().normal(0, 0.01, seed_positions.shape)
                seed_atoms = Atoms(symbols=seed_symbols, positions=seed_positions)
                selected_seed_atom_idxs = [atom.index
                                        for atom, distance_to_c, distance_to_n in zip(seed_atoms, seed_dist_to_center, seed_dist_to_neigh) 
                                        if (atom.symbol in self.species) and (distance_to_c < 10) and (distance_to_c < distance_to_n)]
                
                pocket_dist_to_center = distance_to_center[seed_copy.mol.GetNumAtoms():]
                pocket_dist_to_neigh = distance_to_neigh[seed_copy.mol.GetNumAtoms():]
                assert len(pocket_dist_to_center) == env.pocket_mol.GetNumAtoms()
                pocket_symbols = [atom.GetSymbol() for atom in env.pocket_mol.GetAtoms()]
                pocket_positions = env.pocket_mol.GetConformer().GetPositions()
                pocket_positions = pocket_positions + np.random.default_rng().normal(0, 0.01, pocket_positions.shape)
                pocket_atoms = Atoms(symbols=pocket_symbols, positions=pocket_positions)
                selected_pocket_atom_idxs = [atom.index 
                                            for atom, distance_to_c, distance_to_n in zip(pocket_atoms, pocket_dist_to_center, pocket_dist_to_neigh) 
                                            if (atom.symbol in self.species) and (distance_to_c < 10) and (distance_to_c < distance_to_n)]
                
                # total_atoms = seed_atoms[[env.focal_atom_id]]
                # total_atoms = []
                if len(selected_pocket_atom_idxs) == 0:
                    total_atoms = Atoms(symbols=['C'], positions=[center_pos])
                else:
                    total_atoms = pocket_atoms[selected_pocket_atom_idxs]
                    if len(selected_seed_atom_idxs) > 0:
                        total_atoms += seed_atoms[selected_seed_atom_idxs]
                
                all_atoms.append(total_atoms)
                center_pos_list.append([center_pos])
                
            obs_list = self.soap.create(all_atoms, centers=center_pos_list, n_jobs=16)
            try:
                obs_list = obs_list.squeeze(-2)
            except:
                import pdb;pdb.set_trace()
                
            # obs_list.append(obs)
                
        else:
            for env_i in env_idxs:
                env = self.envs[env_i]
                obs = env.featurize_pocket()
                obs_list.append(obs)
        return obs_list
    
    
    def save_state(self) -> None:
        
        ligand_path = 'ligands.sdf' 
        ligand_writer = Chem.SDWriter(ligand_path)
        
        pocket_path = 'pockets.sdf'  
        pocket_writer = Chem.SDWriter(pocket_path)
        
        native_ligand_path = 'native_ligands.sdf'
        native_writer = Chem.SDWriter(native_ligand_path)
        
        for env in self.envs:
            # mol_h = env.get_clean_fragment(env.seed)
            # ligand_writer.write(mol_h)
            
            ligand_writer.write(env.seed.mol)
            
            clean_pocket = env.get_clean_pocket()
            pocket_writer.write(clean_pocket)
            
            native_ligand = Mol(env.complex.ligand)
            native_writer.write(native_ligand)
                
        ligand_writer.close()
        pocket_writer.close()
        native_writer.close()