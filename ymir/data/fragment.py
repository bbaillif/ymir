import copy

from rdkit import Chem
from rdkit.Chem import Mol, RWMol
from rdkit.Geometry import Point3D
# from rdkit.Chem.rdCIPLabeler import AssignCIPLabels # doesnt work

AtomID = int
AttachPoint = int
Attachments = dict[AtomID, AttachPoint]

class Fragment():

    def __init__(self,
                 mol: Mol,
                 protections: Attachments = None,
                 *args,
                 **kwargs):
        assert isinstance(mol, Mol)
        self.mol = copy.deepcopy(mol)
        # mol_copy = copy.deepcopy(mol) # avoid in place modification
        # super().__init__(mol=mol_copy, *args, **kwargs)
        if protections is not None:
            self.protections = protections
        else:
            self.protections = {}


    def get_attach_points(self,
                          include_protected: bool = False) -> Attachments:
        points = {}
        for atom in self.mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 0:
                isotope = atom.GetIsotope()
                assert isotope != 0
                atom_id = atom.GetIdx()
                points[atom_id] = isotope
        if include_protected:
            for attach_point, label in self.protections.items():
                points[attach_point] = label
        return points
              
                
    def protect(self,
                atom_ids_to_keep: list[int] = [],
                protection_atomic_num: int = 6) -> None:
        
        assert protection_atomic_num in [1, 6]
        
        self.protections: Attachments = {}
        for atom in self.mol.GetAtoms():
            atom_idx: AtomID = atom.GetIdx()
            if not atom_idx in atom_ids_to_keep:
                if atom.GetAtomicNum() == 0:
                    attach_point: AttachPoint = atom.GetIsotope()
                    self.protections[atom_idx] = attach_point
                    if attach_point == 7:
                        atom.SetAtomicNum(8)
                    else:
                        atom.SetAtomicNum(protection_atomic_num)
                    atom.SetIsotope(0)
                    if protection_atomic_num == 6:
                        atom.SetNumExplicitHs(3)
                    else:
                        atom.SetNumExplicitHs(0)
                    # atom.SetNumExplicitHs(0)
        # try:
        #     Chem.SanitizeMol(self.mol)
        #     # Chem.AssignStereochemistryFrom3D(self.mol)
        # except Exception as e:
        #     print(str(e))
        #     import pdb; pdb.set_trace()
    
    
    def unprotect(self,
                  protections: Attachments = None) -> None:
        
        if protections is None:
            protections = self.protections
            
        # Remove Hs that were attached to protections
        rwmol = RWMol(self.mol)
        atoms_to_remove = []
        for atom_idx, attach_point in protections.items():
            attach_atom = rwmol.GetAtomWithIdx(atom_idx)
            neighbors = attach_atom.GetNeighbors()
            for neighbor in neighbors:
                if neighbor.GetAtomicNum() == 1:
                    atoms_to_remove.append(neighbor.GetIdx())
        for atom_idx in reversed(atoms_to_remove):
            rwmol.RemoveAtom(atom_idx)
        new_mol = rwmol.GetMol()
        self.mol = new_mol
            
        for atom_idx, attach_point in protections.items():
            atom = self.mol.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(0)
            atom.SetIsotope(attach_point)
            atom.SetNumExplicitHs(0)
            # atom.SetNoImplicit(True)
        self.protections = {}
        
        try:
            Chem.SanitizeMol(self.mol) # Necessary to set the ExplicitHs number correct for attach points
            Chem.AssignStereochemistryFrom3D(self.mol) # Recompute the stereo for future matching
        except Exception as e:
            print(str(e))
            import pdb; pdb.set_trace()
        
        
    def center(self) -> None:
        conf = self.mol.GetConformer()
        pos = conf.GetPositions()
        centred_pos = pos - pos.mean(axis=0)
        for i, p in enumerate(centred_pos):
            x, y, z = p
            conf.SetAtomPosition(i, Point3D(x, y, z))
            
            
    def set_protections(self,
                        protections: Attachments):
        self.protections = protections
        
    
    @classmethod
    def from_fragment(cls,
                      fragment: 'Fragment') -> 'Fragment':
        return cls(fragment.mol, fragment.protections)
    
    
    def remove_hs(self):
        heavy_atom_ids = [atom.GetIdx() for atom in self.mol.GetAtoms() if atom.GetAtomicNum() != 1]
        self.mol = Chem.RemoveHs(self.mol, updateExplicitCount=True)
        new_protections = {}
        for attach_point, label in self.protections.items():
            new_index = heavy_atom_ids.index(attach_point)
            new_protections[new_index] = label
        self.protections = new_protections
        
        
    def set_attach_label(self,
                         label: int):
        attach_points = self.get_attach_points()
        assert len(attach_points) == 1
        attach_point = list(attach_points.keys())[0]
        attach_atom = self.mol.GetAtomWithIdx(attach_point)
        attach_atom.SetIsotope(label)