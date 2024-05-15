import numpy as np
import torch
import logging
import io
import os

from rdkit import Chem
from rdkit.Chem import Mol
from ase.io import read
from ymir.molecule_builder import add_fragment_to_seed
from ymir.utils.fragment import get_fragments_from_mol, get_neighbor_symbol, center_fragment, find_mappings
from ymir.data.structure.complex import Complex
from torch_geometric.data import Data
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
from ymir.save_state import StateSave, Memory
from ymir.data.structure import GlideProtein
from ymir.metrics.activity import GlideScore
from rdkit.Chem.rdMolAlign import CalcRMS
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
import gzip
import shutil


class FragmentBuilderEnv():
    
    def __init__(self,
                 protected_fragments: list[Fragment],
                 z_table: AtomicNumberTable,
                 max_episode_steps: int = 10,
                 valid_action_masks: dict[int, torch.Tensor] = None,
                 embed_hydrogens: bool = EMBED_HYDROGENS,
                 torsion_angles_deg: list[float] = TORSION_ANGLES_DEG,
                 ) -> None:
        self.protected_fragments = protected_fragments
        self.z_table = z_table
        self.max_episode_steps = max_episode_steps
        self.valid_action_masks = valid_action_masks
        self.embed_hydrogens = embed_hydrogens
        self.torsion_angles_deg = torsion_angles_deg
        
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
              is_real: bool,
              generation_path: str,
              initial_score: float,
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
        
        self.pocket_mol = Mol(self.complex.pocket.mol)
        
        self.transformations = []
        self.actions = []
        
        # random_rotation = Rotation.random()
        # rotate_conformer(self.pocket_mol.GetConformer(), rotation=random_rotation)
        # rotate_conformer(self.seed.to_mol().GetConformer(), rotation=random_rotation)
        # self.transformations.append(random_rotation)
        
        # self.original_seed = Chem.Mol(self.seed)
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        assert self.terminated is False, 'No attachement point in the seed'
        self.truncated = False
        
        assert not self.terminated
        
        self.seed_to_frame()
        
        # observation = self._get_obs()
        info = self._get_info()
        
        has_valid_action = torch.any(self.get_valid_action_mask())
        self.terminated = not has_valid_action
        
        # this should not happen, unless the set of fragments is 
        # not suitable for the given starting situation
        if self.terminated: 
            import pdb;pdb.set_trace()
        
        # self.complex_minimizer = ComplexMinimizer(pocket=self.complex.pocket)
        
        self.new_atom_idxs = range(self.seed.to_mol().GetNumAtoms())
        
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
        return distance
        
        
    def get_new_fragments(self,
                        frag_action: int,
                        sample_rotations: bool) -> list[Fragment]:
        
        fragment_i = frag_action
        protected_fragment = self.protected_fragments[fragment_i]
        
        if sample_rotations:
        
            rotations = []
            for angle in range(-180, 180, 5):
                rotation = Rotation.from_euler('x', angle, degrees=True)
                rotations.append(rotation)
            
            new_fragments = [Fragment.from_fragment(protected_fragment) for _ in rotations]
            for frag, rotation in zip(new_fragments, rotations):
                rotate_conformer(frag.to_mol().GetConformer(), rotation=rotation)
                
        else:
            new_fragments = [Fragment.from_fragment(protected_fragment)]

        return new_fragments
        
        
    def translate_seed(self,
                       added_fragment: Fragment):
        # translate the seed that had the neighbor at 0 to the neighbor at -interatomic_distance
        # Such that the fragment neighbor that is also the seed attach point is (0,0,0)
        distance = self.get_seed_fragment_distance(added_fragment) # hard coded but should be modified
        
        translation = np.array([-distance, 0, 0])
        translate_conformer(self.seed.to_mol().GetConformer(), translation=translation)
        translate_conformer(self.pocket_mol.GetConformer(), translation=translation)
        self.transformations.append(translation)
        
        # Chem.MolToMolFile(self.seed, 'seed_after.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_after.mol')
        # import pdb;pdb.set_trace()
        
        
    def action_to_products(self,
                            frag_action: int,
                            sample_rotations: bool) -> tuple[list[Fragment], list[dict[int, int]]]:
        
        new_fragments = self.get_new_fragments(frag_action,
                                               sample_rotations=sample_rotations)
        new_fragment = new_fragments[0]
        self.translate_seed(added_fragment=new_fragment)
        
        tpls = [add_fragment_to_seed(seed=self.seed,
                                        fragment=new_fragment)
                    for new_fragment in new_fragments]
        products = [tpl[0] for tpl in tpls]
        f2p_mappings = [tpl[2] for tpl in tpls]
        
        # products = [add_fragment_to_seed(seed=self.seed,
        #                                 fragment=new_fragment)[0]
        #             for new_fragment in new_fragments]
        # product_clash = [self.get_pocket_ligand_clash(product.to_mol()) for product in products]
        
        # default_product = products[0]
        # products = [product for product, clash in zip(products, product_clash) if not clash]
        
        # if len(products) == 0:
        #     products = [default_product]
        
        return products, f2p_mappings
        
        if len(products) > 0:
        
            vina_cli = VinaCLI()
            receptor_paths = [self.complex.vina_protein.pdbqt_filepath
                            for _ in products]
            native_ligands = [self.complex.ligand for _ in products]
            frags = [Fragment(product, product.protections) for product in products]
            mols = [self.get_clean_fragment(frag) for frag in frags]
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
            if self.is_real:
                
                n_actions = len(self.actions)
                focal_ideal_position = self.generation_path[n_actions][0]
                frag_copy = Fragment.from_fragment(self.seed)
                for transformation in reversed(self.transformations):
                    if isinstance(transformation, Rotation):
                        rotation_inv = transformation.inv()
                        rotate_conformer(frag_copy.to_mol().GetConformer(), rotation=rotation_inv)
                    else:
                        translation_inv = -transformation
                        translate_conformer(frag_copy.to_mol().GetConformer(), translation=translation_inv)
                attach_points = list(frag_copy.get_attach_points().keys())
                seed_positions = frag_copy.to_mol().GetConformer().GetPositions()
                distances = [euclidean(seed_positions[attach_point], focal_ideal_position)
                            for attach_point in attach_points]
                self.focal_atom_id = attach_points[np.argmin(distances)]
                
            else:
            # self.focal_atom_id = np.random.choice(list(self.attach_points.keys()))
            # to ensure it is deterministic to work with the memory of the BatchEnv
                self.focal_atom_id = list(self.attach_points.keys())[0]
                
            self.seed.protect(atom_ids_to_keep=[self.focal_atom_id])
            assert (len(self.seed.get_attach_points()) == 1)
        return terminated
    
    
    def initiate_step(self,
                      frag_action: int,
                      sample_rotations: bool = True):
        products, f2p_mappings = self.action_to_products(frag_action,
                                           sample_rotations) # seed is deprotected
        return products, f2p_mappings
    
    
    def step(self,
             frag_action: int,
             product: Fragment,
             new_atom_idxs: list[int],
             score: float = 0):
        
        self.actions.append(frag_action)
        self.seed = product
        self.current_score = score
        self.new_atom_idxs = new_atom_idxs
        
        # observation = self._get_obs()
        info = self._get_info()
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        
        n_actions = len(self.actions)
        if (n_actions == self.max_episode_steps) and (not self.terminated): # not terminated but reaching max step size
            self.truncated = True
            # We replace the focal atom with hydrogen (= protect)
            # all other attachment points are already protected
            self.seed.protect() 
        
        if self.terminated or self.truncated:
            assert(all([atom.GetAtomicNum() > 0 for atom in self.seed.to_mol().GetAtoms()]))
        
        # return observation, reward, self.terminated, self.truncated, info
        return self.terminated, self.truncated, info
    
    
    def find_ground_truth(self, twod_frag_action: int):
        try:
            n_torsions = len(self.torsion_angles_deg)
            # twod_frag_action = action % n_torsions
            start = twod_frag_action * n_torsions
            end = (twod_frag_action + 1) * n_torsions
            possible_actions = range(start, end)
            possible_fragments = [self.protected_fragments[action] for action in possible_actions]
            default_products, default_f2p_mappings = self.initiate_step(frag_action=start, sample_rotations=False) # have correct seed translation
            # tpls = [add_fragment_to_seed(seed=self.seed,
            #                             fragment=new_fragment)
            #         for new_fragment in possible_fragments]
            # possible_products = [tpl[0] for tpl in tpls]
            # f2p_mappings = [tpl[2] for tpl in tpls]
            # possible_products = [add_fragment_to_seed(seed=self.seed,
            #                                         fragment=new_fragment)[0]
            #                         for new_fragment in possible_fragments]
            # transformed_native = Mol(self.complex.ligand)
            # for transformation in self.transformations:
            #     if isinstance(transformation, Rotation):
            #         rotate_conformer(transformed_native.GetConformer(), transformation)
            #     else:
            #         translate_conformer(transformed_native.GetConformer(), transformation)
                    
            rmsds = []
            # writer = Chem.SDWriter('products.sdf')
            writer = Chem.SDWriter('possible_fragments.sdf')
            # for product, f2p_mapping in zip(possible_products, f2p_mappings):
            for fragment in possible_fragments:
                # attach_points = product.get_attach_points()
                # n_attaches = len(attach_points)
                # product.protect()
                # product_mol = product.to_mol()
                
                # Chem.SanitizeMol(product_mol)
                # product_mol = Chem.RemoveHs(product_mol)
                # product_mol = Chem.AddHs(product_mol, addCoords=True)
                
                # product_mol = self.get_clean_fragment(product)
                fragment_mol = Fragment.from_fragment(fragment).to_mol()
                for transformation in reversed(self.transformations):
                    if isinstance(transformation, Rotation):
                        rotation_inv = transformation.inv()
                        rotate_conformer(fragment_mol.GetConformer(), rotation=rotation_inv)
                    elif isinstance(transformation, np.ndarray):
                        translation_inv = -transformation
                        translate_conformer(fragment_mol.GetConformer(), translation=translation_inv)
                    else:
                        import pdb;pdb.set_trace()
                writer.write(fragment_mol)
                
                # product_mol = product.to_mol()
                # for transformation in reversed(self.transformations):
                #     if isinstance(transformation, Rotation):
                #         rotation_inv = transformation.inv()
                #         rotate_conformer(product_mol.GetConformer(), rotation=rotation_inv)
                #     elif isinstance(transformation, np.ndarray):
                #         translation_inv = -transformation
                #         translate_conformer(product_mol.GetConformer(), translation=translation_inv)
                #     else:
                #         import pdb;pdb.set_trace()
                # writer.write(product_mol)
                
                # native_noh = Chem.RemoveHs(self.complex.ligand)
                # product_noh = Chem.RemoveHs(product_mol)
                native = Mol(self.complex.ligand)
                # product_new_atom_idxs = list(f2p_mapping.values())
                
                # native_positions = native.GetConformer().GetPositions() #[self.new_atom_idxs]
                # product_positions = product_mol.GetConformer().GetPositions()[product_new_atom_idxs]
                # distance_matrix = cdist(product_positions, native_positions)
                distance_matrix = rdkit_distance_matrix(native, fragment_mol)
                distance_matrix = distance_matrix[:native.GetNumAtoms(), native.GetNumAtoms():]
                
                # z_deg_in_mol = set([(atom.GetAtomicNum(), atom.GetDegree()) 
                #                     for i, atom in enumerate(product_mol.GetAtoms())
                #                     if i in product_new_atom_idxs])
                z_deg_in_mol = set([(atom.GetAtomicNum(), atom.GetDegree()) 
                                    for i, atom in enumerate(fragment_mol.GetAtoms())])
                # if 0 in z_in_mol:
                #     z_in_mol.remove(0)
                min_distances = []
                for z, deg in z_deg_in_mol:
                    fragment_idxs = [i 
                                   for i, atom in enumerate(fragment_mol.GetAtoms())
                                   if atom.GetAtomicNum() == z and atom.GetDegree() == deg]
                    if z in [0, 1]:
                        z_dm = distance_matrix[:,fragment_idxs]
                    else:
                        native_idxs = [i 
                                    for i, atom in enumerate(native.GetAtoms())
                                    if atom.GetAtomicNum() == z and atom.GetDegree() == deg]
                        z_dm = distance_matrix[native_idxs][:, fragment_idxs]
                    z_min_distance = z_dm.min(axis=0)
                    min_distances.extend(z_min_distance)
                
                # min_distance = distance_matrix.min(axis=1)
                # sd = np.square(min_distance)
                sd = np.square(min_distances)
                msd = np.mean(sd)
                min_rmsd = np.sqrt(msd)
                
                # mcs = FindMCS([transformed_native, product_mol])
                # if mcs.numAtoms != (product_mol.GetNumAtoms() - n_attaches):
                #     import pdb;pdb.set_trace()
                # mappings = find_mappings(mcs, transformed_native, product_mol)
                # min_rmsd = np.inf
                # for mapping in mappings:
                #     native_ids = []
                #     product_ids = []
                #     for native_id, product_id in mapping.items():
                #         native_ids.append(native_id)
                #         product_ids.append(product_id)
                #     native_positions = transformed_native.GetConformer().GetPositions()[native_ids]
                #     product_positions = product_mol.GetConformer().GetPositions()[product_ids]
                #     deviation = np.linalg.norm(native_positions - product_positions, axis=1)
                #     sd = np.square(deviation)
                #     msd = np.mean(sd)
                #     rmsd = np.sqrt(msd)
                #     if rmsd < min_rmsd:
                #         min_rmsd = rmsd
                rmsds.append(min_rmsd)
            
            # import pdb;pdb.set_trace()
            
            writer.close()
            Chem.MolToMolFile(self.complex.ligand, 'native.mol')
            Chem.MolToMolFile(self.original_seed.to_mol(), 'original_seed.mol')
            Chem.MolToMolFile(self.get_clean_fragment(self.seed), 'seed.mol')
            
            # if self.complex.ligand.GetProp('_Name') == '4f6w_ligand':
            #     import pdb;pdb.set_trace()
                
            # wrong_names = ['1b8y_ligand', '1y6q_ligand', '4f2w_ligand']
            # wrong_names = []
            # if np.min(rmsds) > 1 and self.complex.ligand.GetProp('_Name') not in wrong_names:
            #     import pdb;pdb.set_trace()
                
            closest_action = np.argmin(rmsds) + start
        except Exception as e:
            print(str(e))
            import pdb;pdb.set_trace()
        return closest_action
    
    
    def step_from_state_save(self,
                             state_save: StateSave):
        self.pocket_mol = Mol(state_save.pocket_mol)
        self.transformations = list(state_save.transformations)
        t = self.step(product=Fragment.from_fragment(state_save.seed), score=float(state_save.score))
        return t
    
    
    # def set_seed_coordinates(self,
    #                          pose_mol):
    #     pose_conformer = pose_mol.GetConformer()
    #     for transformation in self.transformations:
    #         if isinstance(transformation, Rotation):
    #             rotate_conformer(pose_conformer, transformation)
    #         else:
    #             translate_conformer(pose_conformer, transformation)
    #     pose_coordinates = pose_conformer.GetPositions()
    #     seed_conformer = self.seed.to_mol().GetConformer()
    #     for i, new_pos in enumerate(pose_coordinates):
    #         point = Point3D(*new_pos)
    #         seed_conformer.SetAtomPosition(i, point)
    
    
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
                            reverse_transformations: bool = True) -> Mol:
        clean_frag = Fragment.from_fragment(fragment)
        clean_frag.protect()
        mol = clean_frag.to_mol()
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        mol_h = Chem.AddHs(mol, addCoords=True)
        # mol_h = Mol(frag)
        
        if reverse_transformations:
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
    
    
    def get_clean_pocket(self,
                         reverse_transformations: bool = True):
        
        new_mol = Mol(self.pocket_mol)
        if reverse_transformations:
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
        mol = self.get_clean_fragment(self.seed)
        mini_mol = self.complex_minimizer.minimize_ligand(mol, 
                                                        #   distance_constraint=1
                                                          )
        return mini_mol
    
    
    
class BatchEnv():
    
    def __init__(self,
                 envs: list[FragmentBuilderEnv],
                 memory: Memory,
                 scoring_function: str = SCORING_FUNCTION) -> None:
        self.envs = envs
        self.memory = memory
        self.vina_cli = VinaCLI()
        self.reader = PDBQTReader()
        
        assert scoring_function in ['vina', 'glide']
        self.scoring_function = scoring_function
        
        
    def reset(self,
              complexes: list[Complex],
              seeds: list[Fragment],
              initial_scores: list[float],
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
                             
                            #  self.memory
                             )
            batch_info.append(info)
        self.terminateds = [False] * len(self.envs)
        self.truncateds = [False] * len(self.envs)
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
        
        self.seed_idxs = seed_idxs
        
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
             real_data_idxs: list[int],
             ) -> tuple[list[float], list[bool], list[bool], list[dict]]:
        
        assert frag_actions.size()[0] == len(self.ongoing_env_idxs)

        batch_truncated = []
        for env_i, frag_action in zip(self.ongoing_env_idxs, frag_actions):
            env = self.envs[env_i]
            frag_action = int(frag_action)
            
            if env_i in real_data_idxs:
                # step has been initiated already
                new_fragments = env.get_new_fragments(frag_action,
                                                        sample_rotations=False)
                new_fragment = new_fragments[0]
                t = add_fragment_to_seed(seed=env.seed,
                                        fragment=new_fragment)
                product, seed_to_product_mapping, fragment_to_product_mapping = t
                new_atoms_idx = list(fragment_to_product_mapping.values())
                terminated, truncated, info = env.step(frag_action, 
                                                       product,
                                                       new_atoms_idx)
            else:
                products, f2p_mappings = env.initiate_step(frag_action,
                                            sample_rotations=False) # adds actions to env.actions
                product = products[0]
                new_atoms_idx = list(f2p_mappings[0].values())
                
                terminated, truncated, info = env.step(frag_action, 
                                                       product,
                                                       new_atoms_idx)
            
            if truncated:
                terminated = True
                self.truncateds[env_i] = True
            
            if terminated: # we have no attachment points left
                self.terminateds[env_i] = True
            
            batch_truncated.append(truncated)
                
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
        
        return list(self.terminateds), list(self.truncateds)
    
    
    def get_rewards(self,
                    truncateds: list[bool]):
        assert all(self.terminateds)
        rewards = [0 for _ in self.envs]
        scored_env_idxs = []
        receptor_paths = []
        native_ligands = []
        clean_mols = []
        for env_i, env in enumerate(self.envs):
            seed_idx = self.seed_idxs[env_i]
            state_action = (seed_idx, *env.actions)
            if state_action in self.memory:
                rewards[env_i] = self.memory[state_action]
            else:
                scored_env_idxs.append(env_i)
                receptor_paths.append(env.complex.vina_protein.pdbqt_filepath)
                native_ligands.append(env.complex.ligand)
                clean_mols.append(env.get_clean_fragment(env.seed))
                
        if len(scored_env_idxs) > 0:
            vina = VinaCLI()
            scores = vina.get(receptor_paths=receptor_paths,
                            native_ligands=native_ligands,
                            ligands=clean_mols)
            
            for env_i, mol, score in zip(scored_env_idxs, clean_mols, scores):
                env = self.envs[env_i]
                reward = - score
                if mol.GetNumHeavyAtoms() > 50:
                    malus = mol.GetNumHeavyAtoms() - 50
                    reward -= malus
                    
                truncated = truncateds[env_i]
                if truncated:
                    reward -= 10
                    
                rewards[env_i] = reward
                
                seed_idx = self.seed_idxs[env_i]
                state_action = (seed_idx, *env.actions)
                self.memory[state_action] = reward
                
        return rewards
    
    
    def get_rewards_glide(self,
                          truncateds: list[bool],
                          mininplace: bool = True):
        assert all(self.terminateds)
        rewards = [0 for _ in self.envs]
        scored_env_idxs = []
        # receptor_paths = []
        # native_ligands = []
        clean_mols = []
        assert len(set([env.complex.vina_protein.protein_clean_filepath for env in self.envs])) == 1
        for env_i, env in enumerate(self.envs):
            seed_idx = self.seed_idxs[env_i]
            state_action = (seed_idx, *env.actions)
            if state_action in self.memory:
                rewards[env_i] = self.memory[state_action]
            else:
                scored_env_idxs.append(env_i)
                
                # receptor_paths.append(glide_protein.pdbqt_filepath)
                # native_ligands.append(env.complex.ligand)
                clean_mols.append(env.get_clean_fragment(env.seed))
                
        if len(scored_env_idxs) > 0:
            env = self.envs[0]
            glide_protein = GlideProtein(pdb_filepath=env.complex.vina_protein.protein_clean_filepath,
                                            native_ligand=env.complex.ligand)
            glide = GlideScore(glide_protein, mininplace=mininplace)
            scores = glide.get(mols=clean_mols)
            
            with gzip.open('glide_working_dir/glide_scoring_raw.sdfgz', 'rb') as f_in:
                with open('glide_working_dir/glide_scoring_raw.sdf', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            for env_i, mol, score in zip(scored_env_idxs, clean_mols, scores):
                env = self.envs[env_i]
                reward = max(- score, -100)
                if mol.GetNumHeavyAtoms() > 50:
                    malus = mol.GetNumHeavyAtoms() - 50
                    reward -= malus
                    
                truncated = truncateds[env_i]
                if truncated:
                    reward -= 10
                    
                rewards[env_i] = reward
                
                seed_idx = self.seed_idxs[env_i]
                state_action = (seed_idx, *env.actions)
                self.memory[state_action] = reward
                
        return rewards
        
            
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
        for env_i in env_idxs:
            env = self.envs[env_i]
            obs = env.featurize_pocket()
            obs_list.append(obs)
        return obs_list
    
    
    def save_state(self,
                   reverse_transformations: bool = False) -> None:
        
        ligand_path = 'ligands.sdf' 
        ligand_writer = Chem.SDWriter(ligand_path)
        
        pocket_path = 'pockets.sdf'  
        pocket_writer = Chem.SDWriter(pocket_path)
        
        native_ligand_path = 'native_ligands.sdf'
        native_writer = Chem.SDWriter(native_ligand_path)
        
        for env in self.envs:
            mol_h = env.get_clean_fragment(env.seed,
                                           reverse_transformations)
            ligand_writer.write(mol_h)
            
            clean_pocket = env.get_clean_pocket(reverse_transformations)
            pocket_writer.write(clean_pocket)
            
            native_ligand = Mol(env.complex.ligand)
            
            if reverse_transformations:
                # Check if we have effectively returned to the original frame
                ligand_positions = mol_h.GetConformer().GetPositions()
                native_positions = native_ligand.GetConformer().GetPositions()
                distance_matrix = cdist(ligand_positions, native_positions)
                min_distance = distance_matrix.min()
                if min_distance > 0.01:
                    import pdb;pdb.set_trace()
            else:
                # Apply the transformations to the native ligand
                for transformation in env.transformations:
                    if isinstance(transformation, Rotation):
                        rotate_conformer(native_ligand.GetConformer(), rotation=transformation)
                    else:
                        translate_conformer(native_ligand.GetConformer(), translation=transformation)
                    
            native_writer.write(native_ligand)
                
        ligand_writer.close()
        pocket_writer.close()
        native_writer.close()
            
        # with Chem.SDWriter(save_path) as ligand_writer:
        #     for env in self.envs:
        #         mol_h = env.get_clean_fragment(env.seed)
        #         ligand_writer.write(mol_h)
              
        
        # with Chem.SDWriter(pocket_path) as pocket_writer:
        #     for env in self.envs:
        #         clean_pocket = env.get_clean_pocket()
        #         pocket_writer.write(clean_pocket)
                
        # native_ligand_path = 'native_ligands.sdf'
        # with Chem.SDWriter(native_ligand_path) as native_writer:
        #     for env in self.envs:
        #         native_ligand = env.complex.ligand
        #         native_writer.write(native_ligand)
                
        # with Chem.SDWriter('pocket.sdf') as writer:   
        #     writer.write(env.complex.pocket.mol)
        # with Chem.SDWriter('native_ligand.sdf') as writer:   
        #     writer.write(env.complex.ligand)