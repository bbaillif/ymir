import torch

from typing import Sequence, List, Tuple
from .mol_featurizer import MolFeaturizer
from rdkit.Chem import GetPeriodicTable
from torch_geometric.data import Data
from MDAnalysis import Universe, AtomGroup

    
class MDAFeaturizer(MolFeaturizer) :
    """
    Class to transform a molecule into a data point in torch geometric
    Inspired from the QM9 dataset from torch geometric
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.periodic_table = GetPeriodicTable()
    
        
    def featurize_mol(self, 
                      mol: Universe, ) -> List[Data]:
        """
        Transforms all the conformations in the molecule into a list of torch
        geometric data
        
        :param rdkit_mol: Input molecule containing conformations to featurize
        :type rdkit_mol: Mol
        :return: list of data, one for each conformation
        :rtype: List[Data]
        
        """
        
        universe = mol
        
        x = self.encode_atom_features(universe)
        pos = torch.tensor(universe.atoms.positions, dtype=torch.float32)
        
        data = Data(x=x,
                    pos=pos,)
            
        data_list = [data]
        return data_list
    
    def encode_atom_features(self, 
                             universe: Universe) -> torch.tensor :
        """
        Encode the atom features, here only the atomic number (can be modified)
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :return: tensor (n_atoms, 1) storing the atomic numbers
        :rtype: torch.tensor
        
        """
        
        mol_atom_features = []
        atoms: AtomGroup = universe.atoms
        for atom_idx, atom in enumerate(atoms):
            atom_features = []
            atom_symbol = atom.element
            atomic_num = self.periodic_table.GetAtomicNumber(atom_symbol)
            atom_features.append(atomic_num)
            mol_atom_features.append(atom_features)

        x = torch.tensor(mol_atom_features, dtype=torch.float32)
        return x
    
