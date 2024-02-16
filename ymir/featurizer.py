import torch
import numpy as np
from mace.tools.utils import (AtomicNumberTable, 
                              atomic_numbers_to_indices)
from mace.tools.torch_tools import to_one_hot
from torch_geometric.data import Data
from rdkit.Chem import Mol

xyz_coordinate = list[float, float, float]

class Featurizer():
    
    def __init__(self,
                 z_table: AtomicNumberTable) -> None:
        self.z_table = z_table
        
        
    def featurize_mol(self,
                      x: list[int],
                      pos: list[xyz_coordinate]) -> Data:
        
        x = np.array(x)
        x = x.squeeze()
        indices = atomic_numbers_to_indices(x, 
                                            z_table=self.z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(self.z_table),
        )
        
        # x = torch.tensor(one_hot, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        data = Data(x=one_hot,
                    pos=pos)
        
        return data
    
def get_mol_features(mol: Mol,
                    embed_hydrogens: bool = False
                    ) -> tuple[list[int], list[float]]:
    mol_positions = mol.GetConformer().GetPositions()
    x = []
    pos = []
    for atom_i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        embed = True
        if atomic_num == 1:
            if not embed_hydrogens:
                embed = False
                
        if embed:
            atom_pos = mol_positions[atom_i]
            x.append(atomic_num)
            pos.append(atom_pos.tolist())
    
    return x, pos
    