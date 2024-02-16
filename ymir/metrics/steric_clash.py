from ymir.data.structure import Pocket
from rdkit import Chem
from ymir.geometry.geometry_extractor import GeometryExtractor
from rdkit.Chem import Mol
from collections import defaultdict

class StericClash():
    
    def __init__(self, 
                 name: str = 'Steric clash',
                 ) -> None:
        self.geometry_extractor = GeometryExtractor()
        self.clashes = None
        self.valid_pldist_conf_ids = None
        
    
    def get(self,
            pocket_mol: Mol,
            mols: list[Mol]) -> list[float]:
        
        self.clashes = {}
        self.valid_pldist_conf_ids = defaultdict(list)
        self.n_valid = 0
        all_n_clashes = []
        
        for mol in mols:
            ce_clashes = []
            # mols = ce.to_mol_list()
            
            conf_ids = [conf.GetId() for conf in mol.GetConformers()]
            for conf_id in conf_ids:
                
                mol = Mol(mol, confId=conf_id)
                ligand = Chem.AddHs(mol, addCoords=True)
                
                #### Pocket - ligand clashes
                
                complx = Chem.CombineMols(pocket_mol, ligand)
                atoms = [atom for atom in complx.GetAtoms()]
                pocket_atoms = atoms[:pocket_mol.GetNumAtoms()]
                ligand_atoms = atoms[pocket_mol.GetNumAtoms():]
                distance_matrix = Chem.Get3DDistanceMatrix(mol=complx)
                
                n_clashes = 0
            
                for atom1 in pocket_atoms:
                    idx1 = atom1.GetIdx()
                    for atom2 in ligand_atoms:
                        idx2 = atom2.GetIdx()
                        
                        symbol1 = atom1.GetSymbol()
                        symbol2 = atom2.GetSymbol()
                        
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
                            
                        distance = distance_matrix[idx1, idx2]
                        if distance < min_distance:
                            n_clashes = n_clashes + 1
                            invalid_d = {
                            'conf_id': conf_id,
                            'atom_idx1': idx1,
                            'atom_idx2': idx2,
                            'atom_symbol1': symbol1,
                            'atom_symbol2': symbol2,
                            'distance': distance,
                            }
                            ce_clashes.append(invalid_d)
                          
                # Ligand only clashes 
                            
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
                atoms = [atom for atom in ligand.GetAtoms()]
                
                distance_matrix = Chem.Get3DDistanceMatrix(mol=ligand, confId=conf_id)
            
                for i, atom1 in enumerate(atoms):
                    idx1 = atom1.GetIdx()
                    if i != idx1:
                        print('Check atom indices')
                        import pdb;pdb.set_trace()
                    symbol1 = atom1.GetSymbol()
                    for j, atom2 in enumerate(atoms[i+1:]):
                        idx2 = atom2.GetIdx()
                        if i+1+j != idx2:
                            print('Check atom indices')
                            import pdb;pdb.set_trace()
                        symbol2 = atom2.GetSymbol()
                        not_bond = (idx1, idx2) not in bond_idxs
                        not_angle = (idx1, idx2) not in two_hop_idxs
                        not_torsion = (idx1, idx2) not in three_hop_idxs
                        if not_bond and not_angle and not_torsion:
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
                                
                            distance = distance_matrix[idx1, idx2]
                            if distance < min_distance:
                                n_clashes = n_clashes + 1
                                invalid_d = {
                                'conf_id': conf_id,
                                'atom_idx1': idx1,
                                'atom_idx2': idx2,
                                'atom_symbol1': symbol1,
                                'atom_symbol2': symbol2,
                                'distance': distance,
                                }
                                ce_clashes.append(invalid_d)
                    
                all_n_clashes.append(n_clashes)
                    
        return all_n_clashes