import torch
import numpy as np
from rdkit import Chem
from ymir.atomic_num_table import (AtomicNumberTable, 
                                   atomic_numbers_to_indices, 
                                   to_one_hot)
from torch_geometric.data import Data
from rdkit.Chem import Mol
from ymir.params import POCKET_RADIUS, NEIGHBOR_RADIUS
from ymir.data import Fragment
from scipy.spatial.distance import pdist, squareform

xyz_coordinate = list[float, float, float]

class Featurizer():
    
    def __init__(self,
                 z_table: AtomicNumberTable) -> None:
        self.z_table = z_table
    
    def get_mol_features(self,
                         mol: Mol,
                        center_pos: tuple[float, float, float] = None,
                        embed_hydrogens: bool = False,
                        max_radius: float = POCKET_RADIUS,
                        focal_id: int = None,
                        neighbor_radius: float = NEIGHBOR_RADIUS,
                        ) -> tuple[list[int], list[float]]:
        # if not embed_hydrogens:
        #     mol = Chem.RemoveHs(mol)
        mol_positions = mol.GetConformer().GetPositions()
        x = []
        pos = []
        is_focal = []
        center_pos = np.array(center_pos)
        embeded_atom_ids = []
        for atom_id, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            atom_pos = mol_positions[atom_id]
            distance_to_center = np.linalg.norm(center_pos - atom_pos)
            embed = distance_to_center < max_radius
            if atomic_num == 1 and (not embed_hydrogens):
                embed = False
            if embed and (atomic_num in self.z_table.zs):
                embeded_atom_ids.append(atom_id)
                
        distance_matrix = pdist(mol_positions)
        distance_matrix = squareform(distance_matrix)
        all_included_ids = list(embeded_atom_ids)
        for atom_id in embeded_atom_ids:
            dist_to_others = distance_matrix[atom_id]
            within_radius_ids = np.argwhere(dist_to_others < neighbor_radius).flatten().tolist()
            atom = mol.GetAtomWithIdx(atom_id)
            atomic_num = atom.GetAtomicNum()
            assert(any([mol.GetAtomWithIdx(neighbor_id).GetAtomicNum() in self.z_table.zs for neighbor_id in within_radius_ids]))
            for neighbor_id in within_radius_ids:
                atom = mol.GetAtomWithIdx(neighbor_id)
                atomic_num = atom.GetAtomicNum()
                if atomic_num == 1 and (not embed_hydrogens):
                    embed = False
                else:
                    embed = True
                if embed and (atomic_num in self.z_table.zs):
                    all_included_ids.append(neighbor_id)
                
        included_ids = set(all_included_ids)
        for atom_id in included_ids:
            atom = mol.GetAtomWithIdx(atom_id)
            atomic_num = atom.GetAtomicNum()
            atom_pos = mol_positions[atom_id]
            idx = self.z_table.z_to_index(atomic_num)
            x.append(idx)
            # one_hot = [0.0] * len(self.z_table)
            # one_hot[idx] = 1.0
            # x.append(one_hot)
            pos.append(atom_pos.tolist())
            is_focal.append(atom_id == focal_id)
        
        return x, pos, is_focal

    def get_fragment_features(self,
                              fragment: Fragment,
                            center_pos: tuple[float, float, float] = None,
                            embed_hydrogens: bool = False,
                            max_radius: float = POCKET_RADIUS,
                            neighbor_radius: float = NEIGHBOR_RADIUS,):
        
        aps = fragment.get_attach_points()
        assert len(aps) == 1
        focal_id = list(aps.keys())[0]
    
        frag_copy = Fragment.from_fragment(fragment)
        frag_copy.unprotect()
        mol = frag_copy.to_mol()
        x, pos, is_focal = self.get_mol_features(mol=mol, 
                                                center_pos=center_pos,
                                                embed_hydrogens=embed_hydrogens,
                                                max_radius=max_radius,
                                                focal_id=focal_id,
                                                neighbor_radius=neighbor_radius)
        
        return x, pos, is_focal
    