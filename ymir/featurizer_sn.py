import torch
import numpy as np
from ymir.atomic_num_table import (AtomicNumberTable, 
                                   atomic_numbers_to_indices, 
                                   to_one_hot)
from torch_geometric.data import Data
from rdkit.Chem import Mol
from ymir.params import POCKET_RADIUS
from ymir.data import Fragment

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
                        focal_id: int = None
                        ) -> tuple[list[int], list[float]]:
        mol_positions = mol.GetConformer().GetPositions()
        x = []
        pos = []
        is_focal = []
        center_pos = np.array(center_pos)
        for atom_i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            atom_pos = mol_positions[atom_i]
            distance_to_center = np.linalg.norm(center_pos - atom_pos)
            embed = distance_to_center < max_radius
            if atomic_num == 1:
                if not embed_hydrogens:
                    embed = False
            if embed:
                if atomic_num in self.z_table.zs:
                    idx = self.z_table.z_to_index(atomic_num)
                    x.append(idx)
                    # one_hot = [0.0] * len(self.z_table)
                    # one_hot[idx] = 1.0
                    # x.append(one_hot)
                    pos.append(atom_pos.tolist())
                    is_focal.append(atom_i == focal_id)
        
        return x, pos, is_focal

    def get_fragment_features(self,
                              fragment: Fragment,
                            center_pos: tuple[float, float, float] = None,
                            embed_hydrogens: bool = False,
                            max_radius: float = POCKET_RADIUS,):
        
        aps = fragment.get_attach_points()
        assert len(aps) == 1
        focal_id = list(aps.keys())[0]
    
        frag_copy = Fragment(fragment,
                            protections=fragment.protections)
        frag_copy.unprotect()
        x, pos, is_focal = self.get_mol_features(mol=frag_copy, 
                                        center_pos=center_pos,
                                        embed_hydrogens=embed_hydrogens,
                                        max_radius=max_radius,
                                        focal_id=focal_id)
        
        return x, pos, is_focal
    