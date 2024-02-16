import torch

from .mol_featurizer import MolFeaturizer
from rdkit import Chem
from rdkit.Chem import Mol
from torch_geometric.data import Data

    
class RDKitFeaturizer(MolFeaturizer) :
    """
    Class to transform a molecule into a data point in torch geometric
    Inspired from the QM9 dataset from torch geometric
    """
        
    def featurize_mol(self, 
                      mol: Mol, 
                      mol_ids: list[str] = None,
                      embed_hydrogens: bool = False) -> list[Data]:
        """
        Transforms all the conformations in the molecule into a list of torch
        geometric data
        
        :param rdkit_mol: Input molecule containing conformations to featurize
        :type rdkit_mol: Mol
        :param mol_ids: List of identifiers to give each conformation. Length
            must be the same as the number of conformations in the molecule
        :type mol_ids: Sequence
        :param embed_hydrogens: Whether to include the hydrogens in the data
        :type embed_hydrogens: bool
        :return: list of data, one for each conformation
        :rtype: List[Data]
        
        """
        
        rdkit_mol = mol
        
        if mol_ids :
            assert len(mol_ids) == rdkit_mol.GetNumConformers()
        
        data_list = []
        
        if not embed_hydrogens :
            rdkit_mol = Chem.RemoveHs(rdkit_mol)
        
        x = self.encode_atom_features(rdkit_mol)
        mol_bond_features, row, col = self.encode_bond_features(rdkit_mol)

        # Directed graph to undirected
        row, col = row + col, col + row
        edge_index = torch.tensor([row, col])
        edge_attr = torch.tensor(mol_bond_features + mol_bond_features, 
                                 dtype=torch.float32)

        # Sort the edge by source node idx
        perm = (edge_index[0] * rdkit_mol.GetNumAtoms() + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        
        # Make one data per conformer, because it has different positions
        confs = [conf for conf in rdkit_mol.GetConformers()]
        for i, conf in enumerate(confs) : 
            # i can be different than conf_id, i.e. if confs have been removed for a mol
            conf_id = conf.GetId()
            if mol_ids :
                mol_id = mol_ids[i]
            else :
                mol_id = Chem.MolToSmiles(rdkit_mol)
            data = self.conf_to_data(rdkit_mol=rdkit_mol, 
                                     conf_id=conf_id, 
                                     edge_index=edge_index, 
                                     x=x, 
                                     edge_attr=edge_attr,
                                     mol_id=mol_id)
            data_list.append(data)
            
        return data_list
    
    def encode_atom_features(self, 
                             rdkit_mol: Mol) -> torch.tensor :
        """
        Encode the atom features, here only the atomic number (can be modified)
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :return: tensor (n_atoms, 1) storing the atomic numbers
        :rtype: torch.tensor
        
        """
        
        mol_atom_features = []
        for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
            atom_features = []
            atom_features.append(atom.GetAtomicNum())
            mol_atom_features.append(atom_features)

        x = torch.tensor(mol_atom_features, dtype=torch.float32)
        return x
    
    def encode_bond_features(self, 
                             rdkit_mol: Mol) -> tuple[list, list[int], list[int]] :
        """
        Encode the bond features, here none (can be modified)
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :return: tuple storing an empty list, the list of starting atom in bonds
        and the list of ending atom in bonds.
        :rtype: Tuple[list, List[int], List[int]]
        
        """
        mol_bond_features = []
        row = []
        col = []
        for bond in rdkit_mol.GetBonds(): # bonds are undirect, while torch geometric data has directed edge
            bond_features = []
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row.append(start)
            col.append(end)

            mol_bond_features.append(bond_features)
        return mol_bond_features, row, col
    
    def conf_to_data(self, 
                     rdkit_mol: Mol, 
                     conf_id: int, 
                     edge_index: torch.tensor, 
                     x: torch.tensor = None, 
                     edge_attr: torch.tensor = None, 
                     save_mol: bool = False,
                     mol_id: str = None) -> Data: 
        """
        Create a torch geometric Data from a conformation
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :param conf_id: id of the conformation to featurize in the molecule
        :type conf_id: int
        :param edge_index: tensor (n_bonds, 2) containing the start and end of
            each bond in the molecule
        :type edge_index: torch.tensor
        :param x: tensor containing the atomic numbers of each atom in the 
            molecule
        :type x: torch.tensor
        :param edge_attr: tensor to store other atom features (not used)
        :type edge_attr: torch.tensor
        :param save_mol: if True, will save the rdkit molecule as mol_id (not
            recommended, uses space)
        :type save_mol: bool
        :param mol_id: identifier of the conformation (saved in mol_id in Data)
        :type mol_id: str
        :return: single Data containing atomic numbers, positions and bonds for 
            the input conformation
        :rtype: Data
        
        """
        
        conf = rdkit_mol.GetConformer(conf_id)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
            
        data = Data(x=x, 
                    edge_index=edge_index, 
                    pos=pos,)
        
        return data
    
