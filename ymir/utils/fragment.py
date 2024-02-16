from rdkit import Chem
from rdkit.Chem import Mol
from ymir.data import Fragment

def get_neighbor_id_for_atom_id(mol: Mol,
                                atom_id: int) -> int:
    atom = mol.GetAtomWithIdx(atom_id)
    neighbors = atom.GetNeighbors()
    neighbor = neighbors[0]
    neighbor_id = neighbor.GetIdx()
    return neighbor_id


def get_max_num_neighbor(atom_id: int, 
                         mol: Mol):
    
    center_atom = mol.GetAtomWithIdx(atom_id)
    neighbors = center_atom.GetNeighbors()
    assert len(neighbors) > 0, 'There is no neighbors'
    # AssignCIPLabels(mol, atomsToLabel=[atom.GetIdx() for atom in neighbors])
    
    neighbor = neighbors[0]
    max_atom_idx = neighbor.GetIdx()
    max_atomic_num = neighbor.GetAtomicNum()
    max_bond = mol.GetBondBetweenAtoms(atom_id, max_atom_idx)
    max_bond_order = max_bond.GetBondTypeAsDouble()
    
    
    if len(neighbors) > 1:
        for neighbor in neighbors[1:]:
            atomic_num = neighbor.GetAtomicNum()
            neighbor_idx = neighbor.GetIdx()
            if atomic_num > max_atomic_num:
                max_atom_idx = neighbor_idx
                max_atomic_num = atomic_num
                max_bond = mol.GetBondBetweenAtoms(atom_id, max_atom_idx)
                max_bond_order = max_bond.GetBondTypeAsDouble()
            elif atomic_num == max_atomic_num:
                bond = mol.GetBondBetweenAtoms(atom_id, neighbor_idx)
                bond_order = bond.GetBondTypeAsDouble()
                if bond_order > max_bond_order:
                    max_atom_idx = neighbor_idx
                    max_atomic_num = atomic_num
                    max_bond = bond
                    max_bond_order = bond_order
                
    return max_atom_idx


def get_max_num_neighbor_from_original(atom_id: int, 
                                       mol: Mol, 
                                       original_atom_id: int):
    # Because reactions cause a change in atom_ids
    max_atoms = []
    center_atom = mol.GetAtomWithIdx(atom_id)
    neighbors = center_atom.GetNeighbors()
    prop_name = f'_IsMax_{original_atom_id}'
    for neighbor in neighbors:
        if neighbor.HasProp(prop_name):
            max_atoms.append(neighbor.GetIdx())
    if len(max_atoms) > 1:
        raise Exception('too much atom fitting description')
    elif len(max_atoms) == 1:
        return max_atoms[0]
    else:
        # print(f'{prop_name} was not defined')
        return None
    
    
def get_fragments_from_mol(mol: Mol) -> list[Fragment]:
    pieces = Chem.BRICS.BreakBRICSBonds(mol)
    frags = Chem.GetMolFrags(pieces, asMols=True)
    fragments = [Fragment(mol) for mol in frags]
    return fragments


def get_unique_fragments_from_mols(mols: list[Mol]) -> list[Fragment]:
    
    unique_fragments = []
    all_smiles = []
    for mol in mols:
        fragments = get_fragments_from_mol(mol)
        if len(fragments) > 1:
            for i, frag in enumerate(fragments):
                attach_points = frag.get_attach_points()
                if len(attach_points) > 0:
                    smiles = Chem.MolToSmiles(frag, 
                                            allHsExplicit=True)
                    new_name = mol.GetProp('_Name') + f'_{i}'
                    frag.SetProp('_Name', 
                                new_name)
                    if not smiles in all_smiles:
                        all_smiles.append(smiles)
                        unique_fragments.append(frag)
                
    return unique_fragments

# class ProtectedFragment(Fragment):
    
#     def __init__(self,
#                  fragment: Fragment,
#                  protections: Attachments,
#                  atom_ids_to_keep: list[int] = [],
#                  *args,
#                  **kwargs):
#         fragment_copy = copy.deepcopy(fragment) # avoid inplace modifications
#         super().__init__(self, mol=fragment_copy, *args, **kwargs)
#         self.atom_ids_to_keep
#         self.protect_fragment()
#         self._protections = protections
    
#     @property
#     def protections(self):
#         return self._protections

#     def protect_fragment(self: Fragment,
#                         atom_ids_to_keep: list[int] = []) -> ProtectedFragment:
        
#         fragment_copy = copy.deepcopy(fragment) # avoid inplace modifications
        
#         protections: Attachments = {}
#         for atom in fragment_copy.GetAtoms():
#             atom_idx: AtomID = atom.GetIdx()
#             if not atom_idx in atom_ids_to_keep:
#                 if atom.GetAtomicNum() == 0:
#                     attach_point: AttachPoint = atom.GetIsotope()
#                     protections[atom_idx] = attach_point
#                     atom.SetAtomicNum(1)
#                     atom.SetIsotope(0)
                    
#         protected_fragment = ProtectedFragment(fragment=fragment_copy,
#                                             protections=protections)
#         return protected_fragment
            

# def deprotect_fragment(protected_fragment: ProtectedFragment) -> Fragment:
    
#     fragment = protected_fragment.fragment
#     protections = protected_fragment.protections
#     for atom_idx, attach_point in protections.items():
#         atom = fragment.GetAtomWithIdx(atom_idx)
#         atom.SetAtomicNum(0)
#         atom.SetIsotope(attach_point)
        
#     return fragment
        
        
# def center_fragment(fragment: Fragment):
#     conf = fragment.GetConformer()
#     pos = conf.GetPositions()
#     centred_pos = pos - pos.mean(axis=0)
#     for i, p in enumerate(centred_pos):
#         x, y, z = p
#         conf.SetAtomPosition(i, Point3D(x, y, z))