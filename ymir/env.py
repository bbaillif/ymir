import numpy as np
import torch
import logging

from rdkit import Chem
from rdkit.Chem import Mol
from ymir.molecule_builder import add_fragment_to_seed
from ymir.utils.fragment import get_fragments_from_mol
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
from ymir.params import EMBED_HYDROGENS, TORSION_ANGLES_DEG
from ymir.featurizer_sn import Featurizer


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
            assert mask.size()[-1] == self.action_dim

        self.seed: Fragment = None
        self.fragment: Fragment = None
        
        self.geometry_extractor = GeometryExtractor()
        self.featurizer = Featurizer(z_table=self.z_table)
        
    @property
    def action_dim(self) -> int:
        return self._action_dim
        
        
    def seed_to_frame(self):
        for atom in self.seed.GetAtoms():
            if atom.GetAtomicNum() == 0:
                attach_point = atom
                break
        
        # Chem.MolToMolFile(self.seed, 'seed_before.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_before.mol')
        
        # Align the neighbor ---> attach point vector to the x axis: (0,0,0) ---> (1,0,0)
        # Then translate such that the neighbor is (0,0,0)
        neighbor = attach_point.GetNeighbors()[0]
        neighbor_id = neighbor.GetIdx()
        attach_id = attach_point.GetIdx()
        positions = self.seed.GetConformer().GetPositions()
        # neighbor_attach = positions[[neighbor_id, attach_id]]
        # distance = euclidean(neighbor_attach[0], neighbor_attach[1])
        # x_axis_vector = np.array([[0,0,0], [distance,0,0]])
        neighbor_pos = positions[neighbor_id]
        attach_pos = positions[attach_id]
        neighbor_attach = attach_pos - neighbor_pos
        distance = euclidean(attach_pos, neighbor_pos)
        x_axis_vector = np.array([distance, 0, 0])
        # import pdb;pdb.set_trace()
        rotation, rssd = Rotation.align_vectors(a=x_axis_vector.reshape(-1, 3), b=neighbor_attach.reshape(-1, 3))
        rotate_conformer(conformer=self.seed.GetConformer(),
                         rotation=rotation)
        rotate_conformer(conformer=self.pocket_mol.GetConformer(),
                         rotation=rotation)
        
        positions = self.seed.GetConformer().GetPositions()
        neighbor_pos = positions[neighbor_id]
        translation = -neighbor_pos
        translate_conformer(conformer=self.seed.GetConformer(),
                            translation=translation)
        translate_conformer(conformer=self.pocket_mol.GetConformer(),
                            translation=translation)
        
        self.transformations.append(rotation)
        self.transformations.append(translation)
        
        Chem.MolToMolFile(self.seed, 'seed_after.mol')
        Chem.MolToMolFile(self.pocket_mol, 'pocket_after.mol')
        
        
    def get_seed_fragment_distance(self,
                                   fragment: Fragment):
        # Get fragment attach neighbor (which is the seed attach point)
        # Get Seed attach neighbor (which is the fragment attach point)
        # Compute the ideal length between the two atoms
        
        return 1.45 # hard coded but should be modified
        
        
    def get_new_fragment(self,
                        frag_action: int,
                        vector_action: torch.Tensor):
        fragment_i = frag_action
        assert fragment_i < self.action_dim, 'Invalid action'
        protected_fragment = self.protected_fragments[fragment_i]
        
        # rotation along the X axis
        new_fragment = Fragment(protected_fragment,
                                protections=protected_fragment.protections)
        
        # x_axis_vector = np.array([-1, 0, 0])
        # y_axis_vector = np.array([0, 1, 0])
        # action_vector = vector_action # should have norm = 1 on xyz
        # action_vector[0] = 0
        # neighbor_attach = np.array([-1, 0, 0])
        # The fragment originates at (0, 0, 0)
        # We imagine the original orientation vector of the fragment is (0, 1, 0)
        # we will rotate to align this orientation vector with the action vector
        
        base_yz_vector = torch.tensor([1.0, 0.0])
        action_yz_vector = vector_action[1:] # we don't need x, as we already are fixed on the x axis
        denominator = base_yz_vector.norm() * action_yz_vector.norm()
        cos_theta = torch.dot(base_yz_vector, action_yz_vector) / denominator
        theta = torch.arccos(cos_theta)
        sin_theta = torch.sin(theta)
        
        # Rotation on the yz plane
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cos_theta, -sin_theta],
                                    [0, sin_theta, cos_theta]])
        
        rotation = Rotation.from_matrix(rotation_matrix)
        
        rotate_conformer(new_fragment.GetConformer(), rotation=rotation)
        
        Chem.MolToMolFile(new_fragment, 'fragment_after.mol')
        
        return new_fragment
        
        
    def translate_seed(self,
                       fragment: Fragment):
        # translate the seed that had the neighbor at 0 to the neighbor at -interatomic_distance
        # Such that the fragment neighbor that is also the seed attach point is (0,0,0)
        distance = self.get_seed_fragment_distance(fragment) # hard coded but should be modified
        
        translation = np.array([-distance, 0, 0])
        translate_conformer(self.seed.GetConformer(), translation=translation)
        translate_conformer(self.pocket_mol.GetConformer(), translation=translation)
        self.transformations.append(translation)
        
        # Chem.MolToMolFile(self.seed, 'seed_after.mol')
        # Chem.MolToMolFile(self.pocket_mol, 'pocket_after.mol')
        # import pdb;pdb.set_trace()
        
        
    def action_to_fragment_build(self,
                                 frag_action: int,
                                 vector_action: list[float]):
        
        new_fragment = self.get_new_fragment(frag_action, vector_action)
        self.translate_seed(fragment=new_fragment)
        
        product = add_fragment_to_seed(seed=self.seed,
                                        fragment=new_fragment)
        
        self.seed = product
        
    
    def _get_obs(self):
        data = self.featurize_pocket()
        obs = data
        return obs
    
    
    def _get_info(self):
        return {'protein_path': self.complex.vina_protein,
                'seed_i': self.seed_i}
    
    
    def featurize_pocket(self) -> Data:
        center_pos = [0, 0, 0]
        ligand_x, ligand_pos = self.featurizer.get_fragment_features(fragment=self.seed, 
                                                                    embed_hydrogens=self.embed_hydrogens,
                                                                    center_pos=center_pos)
        protein_x, protein_pos = self.featurizer.get_mol_features(mol=self.pocket_mol,
                                                                embed_hydrogens=self.embed_hydrogens,
                                                                center_pos=center_pos)
        
        pocket_x = protein_x + ligand_x
        pocket_pos = protein_pos + ligand_pos
        
        mol_id = [0] * len(protein_x) + [1] * len(ligand_x)
        
        x = torch.tensor(pocket_x, dtype=torch.long)
        pos = torch.tensor(pocket_pos, dtype=torch.float)
        mol_id = torch.tensor(mol_id)
        data = Data(x=x,
                    pos=pos,
                    mol_id=mol_id)

        return data
    
    
    def reset(self, 
              complx: Complex) -> tuple[Data, dict[str, Any]]:
        
        self.complex = complx
        self.pocket_mol = Mol(self.complex.pocket.mol)
        fragments = get_fragments_from_mol(self.complex.ligand)
        self.seed_i = np.random.choice(len(fragments))
        # self.seed_i = 0
        self.seed = fragments[self.seed_i]
        
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
             vector_action: list[float]):
        
        self.actions.append((frag_action, vector_action))
        n_actions = len(self.actions)
        
        self.action_to_fragment_build(frag_action, vector_action) # seed is deprotected
        
        self.terminated = self.set_focal_atom_id() # seed is protected
        
        reward = 0 # reward is handled in batch, once all envs are done
        
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
    
    
    def get_clashes(self) -> list[bool]:
        
        master_mol = Chem.CombineMols(self.seed, self.pocket_mol)
        for fragment in self.protected_fragments:
            master_mol = Chem.CombineMols(master_mol, fragment)
            
        from timeit import default_timer as timer
        start = timer()
        distance_matrix = Chem.Get3DDistanceMatrix(mol=master_mol)
        print(timer() - start)
        
        import pdb;pdb.set_trace()
        
        return [False] * self.n_fragments
    
    
    def get_clashes2(self) -> list[bool]:
        
        for atom in self.seed.GetAtoms():
            if atom.GetAtomicNum() == 0:
                seed_attach_atom = atom
                break
        n_seed_atoms = self.seed.GetNumAtoms()
        
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
        
        all_has_clashes = []
        for fragment in tqdm(self.protected_fragments):
            
            assert fragment.GetNumConformers() == 1
            
            #### Pocket - ligand clashes
            
            # pf = Chem.CombineMols(self.pocket_mol, fragment)
            # pfs = Chem.CombineMols(pf, self.seed)
            # atoms = [atom for atom in pfs.GetAtoms()]
            # n_pocket_atoms = self.pocket_mol.GetNumAtoms()
            # n_fragment_atoms = fragment.GetNumAtoms()
            # pocket_atoms = atoms[:n_pocket_atoms]
            # fragment_atoms = atoms[n_pocket_atoms:n_pocket_atoms+n_fragment_atoms]
            # seed_atoms = atoms[n_pocket_atoms+n_fragment_atoms:]
            # distance_matrix = Chem.Get3DDistanceMatrix(mol=pfs)
            # pf_distance_matrix = distance_matrix[:n_pocket_atoms, n_pocket_atoms:n_pocket_atoms+n_fragment_atoms]
            
            # TOO SLOW
            pocket_pos = self.pocket_mol.GetConformer().GetPositions()
            pocket_atoms = self.pocket_mol.GetAtoms()
            # n_pocket_atoms = len(pocket_atoms)
            fragment_pos = fragment.GetConformer().GetPositions()
            fragment_atoms = fragment.GetAtoms()
            # n_fragment_atoms = len(fragment_atoms)
            # distance_matrix = cdist(pocket_pos, fragment_pos)
            
            margin = 2.7
            min_fragment_pos = fragment_pos.min(axis=0)
            max_fragment_pos = fragment_pos.max(axis=0)
            min_fragment_pos = min_fragment_pos - margin
            max_fragment_pos = max_fragment_pos + margin
            beyond_fragment_min = (pocket_pos < min_fragment_pos).any(axis=1)
            beyond_fragment_max = (pocket_pos > max_fragment_pos).any(axis=1)
            beyond_fragment = np.logical_or(beyond_fragment_min, beyond_fragment_max)
            close_to_fragment = np.logical_not(beyond_fragment)
            indices_close = np.nonzero(close_to_fragment)[0]
            
            if len(indices_close) > 0:
                pocket_pos = pocket_pos[indices_close]
                pocket_atoms = [atom 
                                for i, atom in enumerate(pocket_atoms) 
                                if i in indices_close]
                # pf_distance_matrix = metrics.pairwise_distances(pocket_pos, fragment_pos)
                pf_distance_matrix = cdist(pocket_pos, fragment_pos)
            else:
                pocket_atoms = []
                pf_distance_matrix = []
                
            # import pdb;pdb.set_trace()
            
            has_clash = False
        
            for idx1, atom1 in enumerate(pocket_atoms):
                for idx2, atom2 in enumerate(fragment_atoms):
                    
                    symbol1 = atom1.GetSymbol()
                    symbol2 = atom2.GetSymbol()
                    if symbol2 == 'R':
                        symbol2 = 'C' # we mimic that the fragment will bind to a Carbon
                        
                    if symbol1 > symbol2: # sort symbols
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
                        
                    distance = pf_distance_matrix[idx1, idx2]
                    if distance < min_distance:
                        has_clash = True
                        break
                    
                if has_clash:
                    break
                        
            # Seed - Fragment clashes
                      
            if not has_clash:
                      
                seed_attach_atom_id = seed_attach_atom.GetIdx()
                seed_neighbor_atom_id = seed_attach_atom.GetNeighbors()[0].GetIdx()
                      
                for atom in fragment.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        fragment_attach_atom = atom
                        break
                      
                fragment_attach_atom_id = fragment_attach_atom.GetIdx()
                fragment_neighbor_atom_id = fragment_attach_atom.GetNeighbors()[0].GetIdx()
                
                # fragment_attach_atom_id = fragment_attach_atom_id + n_seed_atoms
                # fragment_neighbor_atom_id = fragment_neighbor_atom_id + n_seed_atoms
                
                # fs_distance_matrix = distance_matrix[n_pocket_atoms:n_pocket_atoms+n_fragment_atoms, n_pocket_atoms+n_fragment_atoms:]
                
                seed_pos = self.seed.GetConformer().GetPositions()
                seed_atoms = self.pocket_mol.GetAtoms()
                
                min_fragment_pos = fragment_pos.min(axis=0)
                max_fragment_pos = fragment_pos.max(axis=0)
                beyond_fragment_min = (seed_pos < min_fragment_pos).any(axis=1)
                beyond_fragment_max = (seed_pos > max_fragment_pos).any(axis=1)
                beyond_fragment = np.logical_or(beyond_fragment_min, beyond_fragment_max)
                close_to_fragment = np.logical_not(beyond_fragment)
                indices_close = np.nonzero(close_to_fragment)[0]
                
                if len(indices_close) > 0:
                
                    seed_pos = seed_pos[indices_close]
                    seed_atoms = [atom 
                                    for i, atom in enumerate(seed_atoms) 
                                    if i in indices_close]
                    
                    # fs_distance_matrix = metrics.pairwise_distances(fragment_pos, seed_pos)
                    fs_distance_matrix = cdist(fragment_pos, seed_pos)
                    
                else:
                    seed_atoms = []
                    fs_distance_matrix = []
                
                # import pdb;pdb.set_trace()

                for idx1, atom1 in enumerate(fragment_atoms):
                    if idx1 not in [fragment_neighbor_atom_id, fragment_attach_atom_id]:
                        for idx2, atom2 in enumerate(seed_atoms):
                            if idx2 not in [seed_neighbor_atom_id, seed_attach_atom_id]:
                            
                                symbol1 = atom1.GetSymbol()
                                symbol2 = atom2.GetSymbol()
                                if symbol2 == 'R':
                                    symbol2 = 'C' # we mimic that the fragment will bind to a Carbon
                                    
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
                                    
                                distance = fs_distance_matrix[idx1, idx2]
                                if distance < min_distance:
                                    has_clash = True
                                    break
                        
                    if has_clash:
                        break
                        
            all_has_clashes.append(has_clash)
                    
        return all_has_clashes
    
    
    def get_pocket_seed_clash(self):
        
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
        
        ps = Chem.CombineMols(self.pocket_mol, self.seed)
        atoms = [atom for atom in ps.GetAtoms()]
        n_pocket_atoms = self.pocket_mol.GetNumAtoms()
        pocket_atoms = atoms[:n_pocket_atoms]
        seed_atoms = atoms[n_pocket_atoms:]
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
            for idx2, atom2 in enumerate(seed_atoms):
                
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

            bonds = self.geometry_extractor.get_bonds(self.seed)
            bond_idxs = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        for bond in bonds]
            bond_idxs = [t 
                        if t[0] < t[1] 
                        else (t[1], t[0])
                        for t in bond_idxs ]
            angle_idxs = self.geometry_extractor.get_angles_atom_ids(self.seed)
            two_hop_idxs = [(t[0], t[2])
                                    if t[0] < t[2]
                                    else (t[2], t[0])
                                    for t in angle_idxs]
            torsion_idxs = self.geometry_extractor.get_torsions_atom_ids(self.seed)
            three_hop_idxs = [(t[0], t[3])
                                if t[0] < t[3]
                                else (t[3], t[0])
                                for t in torsion_idxs]

            for idx1, atom1 in enumerate(seed_atoms):
                for idx2, atom2 in enumerate(seed_atoms[idx1+1:]):
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
    
    def get_clean_mol(self):
        frag = Fragment(self.seed)
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
    
    
class BatchEnv():
    
    def __init__(self,
                 envs: list[FragmentBuilderEnv]) -> None:
        self.envs = envs
        
        
    def reset(self,
              complexes: list[Complex],
              scorer) -> tuple[list[Data], list[dict[str, Any]]]:
        # batch_obs: list[Data] = []
        assert len(complexes) == len(self.envs)
        batch_info: list[dict[str, Any]] = []
        for env, complx in zip(self.envs, complexes):
            # obs, info = env.reset(complx)
            info = env.reset(complx)
            # batch_obs.append(obs)
            batch_info.append(info)
        # batch_obs = Batch.from_data_list(batch_obs)
        # return batch_obs, batch_info
        self.terminateds = [False] * len(self.envs)
        self.ongoing_env_idxs = list(range(len(self.envs)))
        
        # Chem.MolToMolFile(complexes[0].pocket.mol, 'original_pocket.mol')
        
        self.scorer = scorer
        # self.reward_function = DockingBatchRewards(complx, scorer)
        
        return batch_info
    
    
    def get_valid_action_mask(self) -> torch.Tensor:
        masks = []
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            valid_action_mask = env.valid_action_mask # the get_valid_action_mask() is called earlier
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
    
    
    def get_rewards(self,
                    max_num_heavy_atoms: int = 50,
                    clash_penalty: float = -10.0) -> list[float]:
        
        rewards = []
        for env_i in self.ongoing_env_idxs:
            env = self.envs[env_i]
            terminated = self.terminateds[env_i]
            mol = env.get_clean_mol()
            # n_clashes = env.complex.clash_detector.get([mol])
            # n_clash = n_clashes[0]
            has_clash = env.get_pocket_seed_clash()
            if has_clash: # we penalize clash
                reward = clash_penalty
            elif terminated: # if no clash, we score the ligand is construction is finished
                # scores = env.vina_score.get([mol])
                scores = self.scorer.get([mol])
                score = scores[0]
                reward = -score # we want to maximize the reward, so minimize glide score
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
             vector_actions: torch.Tensor
             ) -> tuple[list[float], list[bool], list[bool], list[dict]]:
        
        assert frag_actions.size()[0] == len(self.ongoing_env_idxs)
        assert vector_actions.size()[0] == len(self.ongoing_env_idxs)

        batch_truncated = []
        batch_info = []
        mols = []
        for env_i, frag_action, vector_action in zip(self.ongoing_env_idxs, frag_actions, vector_actions):
            env = self.envs[env_i]
            _, terminated, truncated, info = env.step(frag_action, vector_action)
            
            if terminated: # we have no attachment points left
                self.terminateds[env_i] = True
            
            batch_truncated.append(truncated)
            batch_info.append(info)
            
            mols.append(env.get_clean_mol())
            
        batch_rewards = self.get_rewards()
        
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
        with Chem.SDWriter(save_path) as ligand_writer:
            for env in self.envs:
                mol_h = env.get_clean_mol()
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