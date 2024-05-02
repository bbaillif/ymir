import copy

from rdkit.Chem import Mol
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


    def get_attach_points(self) -> Attachments:
        points = {}
        for atom in self.mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 0:
                isotope = atom.GetIsotope()
                assert isotope != 0
                atom_id = atom.GetIdx()
                points[atom_id] = isotope
        return points
              
                
    def protect(self,
                atom_ids_to_keep: list[int] = []) -> None:
        self.protections: Attachments = {}
        for atom in self.mol.GetAtoms():
            atom_idx: AtomID = atom.GetIdx()
            if not atom_idx in atom_ids_to_keep:
                if atom.GetAtomicNum() == 0:
                    attach_point: AttachPoint = atom.GetIsotope()
                    self.protections[atom_idx] = attach_point
                    atom.SetAtomicNum(1)
                    atom.SetIsotope(0)
    
    
    def unprotect(self,
                  protections: Attachments = None) -> None:
        if protections is None:
            protections = self.protections
        for atom_idx, attach_point in protections.items():
            atom = self.mol.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(0)
            atom.SetIsotope(attach_point)
        self.protections = {}
        
        
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
        
        
    def to_mol(self,
               protect: bool = False) -> Mol:
        if protect:
            self.protect()
        return self.mol
    
    @classmethod
    def from_fragment(cls,
                      fragment: 'Fragment') -> 'Fragment':
        return cls(fragment.mol, fragment.protections)