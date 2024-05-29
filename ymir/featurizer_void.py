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
from scipy.spatial.distance import pdist, squareform, cdist

xyz_coordinate = list[float, float, float]

class Featurizer():
    
    def __init__(self,
                 z_table: AtomicNumberTable) -> None:
        self.z_table = z_table
    
    def get_mol_features(self,
                         mol: Mol,
                         dummy_points: np.ndarray,
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
        
        if len(dummy_points) == 0:
            dummy_points = np.array([center_pos])
        
        is_heavy = np.array([atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()])
        heavy_atom_ids = np.where(is_heavy)[0].tolist()
        heavy_pos = mol_positions[is_heavy]
        
        distance_matrix = cdist(heavy_pos, dummy_points)
        min_distance = np.min(distance_matrix, axis=1)
        embeded_heavy_atom_ids = np.argwhere(min_distance < neighbor_radius).flatten().tolist()
        
        # closest_atom_ids = np.argsort(distance_matrix, axis=1)
        # n_closest_per_atom = 3
        # embeded_heavy_atom_ids = set(closest_atom_ids[:, :n_closest_per_atom].flatten().tolist())
        
        for heavy_atom_id in embeded_heavy_atom_ids:
            atom_id = heavy_atom_ids[heavy_atom_id]
            atom = mol.GetAtomWithIdx(atom_id)
            atomic_num = atom.GetAtomicNum()
            # embed = atom_id != focal_id
            embed = True
            # if atomic_num == 1 and (not embed_hydrogens):
            #     embed = False
            if embed and (atomic_num in self.z_table.zs):
                atom_pos = mol_positions[atom_id]
                idx = self.z_table.z_to_index(atomic_num)
                x.append(idx)
                pos.append(atom_pos.tolist())
                is_focal.append(atom_id == focal_id)
        
        return x, pos, is_focal

    def get_fragment_features(self,
                              fragment: Fragment,
                              dummy_points: np.ndarray,
                            center_pos: tuple[float, float, float] = None,
                            embed_hydrogens: bool = False,
                            max_radius: float = POCKET_RADIUS,
                            neighbor_radius: float = NEIGHBOR_RADIUS,):
        
        aps = fragment.get_attach_points()
        assert len(aps) == 1
        focal_id = list(aps.keys())[0]
    
        frag_copy = Fragment.from_fragment(fragment)
        frag_copy.unprotect()
        mol = frag_copy.mol
        x, pos, is_focal = self.get_mol_features(mol=mol, 
                                                 dummy_points=dummy_points,
                                                center_pos=center_pos,
                                                embed_hydrogens=embed_hydrogens,
                                                max_radius=max_radius,
                                                focal_id=focal_id,
                                                neighbor_radius=neighbor_radius)
        
        return x, pos, is_focal
    