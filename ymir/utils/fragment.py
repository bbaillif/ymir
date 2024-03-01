import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Mol
from ymir.data import Fragment
from ymir.data.structure import Complex
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean
from ymir.utils.spatial import (rotate_conformer, 
                                translate_conformer)
from ymir.molecule_builder import potential_reactions
from typing import Union, NamedTuple
from tqdm import tqdm

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


def get_construct(ligand: Mol,
                removed_fragment: Fragment) -> Fragment:
    
    ligand_positions = ligand.GetConformer().GetPositions()
    fragment_positions = removed_fragment.GetConformer().GetPositions()
    fragment_aps = removed_fragment.get_attach_points()
    fragment_ap_idx = list(fragment_aps.keys())[0]
    fragment_ap_atom = removed_fragment.GetAtomWithIdx(fragment_ap_idx)
    fragment_ap_neighbors = fragment_ap_atom.GetNeighbors()
    assert len(fragment_ap_neighbors) == 1
    fragment_neigh_idx = fragment_ap_neighbors[0].GetIdx()
    
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(fragment_positions, ligand_positions)
    min_dists_idx = distance_matrix.argmin(axis=1)
    
    ligand_ap_idx = int(min_dists_idx[fragment_ap_idx])
    ligand_neigh_idx = int(min_dists_idx[fragment_neigh_idx])
    
    brics_bonds = Chem.BRICS.FindBRICSBonds(ligand)
    broken = False
    for bond in brics_bonds:
        t_bond, t_labels = bond
        t1 = (ligand_ap_idx, ligand_neigh_idx)
        t2 = (ligand_neigh_idx, ligand_ap_idx)
        if t_bond == t1 or t_bond == t2 :
            new_mol = Chem.BRICS.BreakBRICSBonds(ligand, bonds=[bond])
            broken = True
            break
    if not broken:
        import pdb;pdb.set_trace()
    
    new_frags = Chem.GetMolFrags(new_mol, asMols=True)
    assert len(new_frags) == 2
    frag1, frag2 = new_frags
    if Chem.MolToSmiles(frag1) == Chem.MolToSmiles(removed_fragment):
        construct = Fragment(frag2)
    else:
        if Chem.MolToSmiles(frag2) != Chem.MolToSmiles(removed_fragment):
            if Chem.MolToSmiles(frag1, isomericSmiles=False) == Chem.MolToSmiles(removed_fragment, isomericSmiles=False):
                construct = Fragment(frag2)
            else:
                if Chem.MolToSmiles(frag2, isomericSmiles=False) != Chem.MolToSmiles(removed_fragment, isomericSmiles=False):
                    import pdb;pdb.set_trace()
        # assert Chem.MolToSmiles(frag2) == Chem.MolToSmiles(self.removed_fragment)
                construct = Fragment(frag1)
        else:
            construct = Fragment(frag1)
            
    return construct


class ConstructionSeed(NamedTuple): 
    construct: Fragment
    removed_fragment: Fragment


def get_seeds(ligand: Mol) -> list[ConstructionSeed]:
    fragments = get_fragments_from_mol(ligand)
    
    fragments_1ap: list[Fragment] = []
    for fragment in fragments:
        aps = fragment.get_attach_points()
        if len(aps) == 1:
            fragments_1ap.append(fragment)
            
    seeds = []
    for removed_fragment in fragments_1ap :
        construct = get_construct(ligand, removed_fragment)
        construction_seed = ConstructionSeed(construct, removed_fragment)
        seeds.append(construction_seed)
            
    return seeds


# Align the attach point ---> neighbor vector to the x axis: (0,0,0) ---> (1,0,0)
# Then translate such that the neighbor is (0,0,0)
def center_fragments(protected_fragments: list[Fragment]):
    for fragment in protected_fragments:
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() == 0:
                attach_point = atom
                break
        neighbor = attach_point.GetNeighbors()[0]
        neighbor_id = neighbor.GetIdx()
        attach_id = attach_point.GetIdx()
        positions = fragment.GetConformer().GetPositions()
        # neighbor_attach = positions[[neighbor_id, attach_id]]
        # distance = euclidean(neighbor_attach[0], neighbor_attach[1])
        # x_axis_vector = np.array([[0,0,0], [distance,0,0]])
        neighbor_pos = positions[neighbor_id]
        attach_pos = positions[attach_id]
        attach_neighbor = neighbor_pos - attach_pos
        distance = euclidean(neighbor_pos, attach_pos)
        x_axis_vector = np.array([distance, 0, 0])
        # import pdb;pdb.set_trace()
        rotation, rssd = Rotation.align_vectors(a=x_axis_vector.reshape(-1, 3), b=attach_neighbor.reshape(-1, 3))
        rotate_conformer(conformer=fragment.GetConformer(),
                            rotation=rotation)
        
        positions = fragment.GetConformer().GetPositions()
        neighbor_pos = positions[neighbor_id]
        translation = -neighbor_pos
        translate_conformer(conformer=fragment.GetConformer(),
                            translation=translation)


def get_masks(final_fragments: list[Fragment]):
    fragment_attach_labels = []
    for act_i, fragment in enumerate(final_fragments):
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() == 0:
                attach_label = atom.GetIsotope()
                break
        fragment_attach_labels.append(attach_label)

    valid_action_masks: dict[int, list[bool]] = {}
    for attach_label_1, d_potential_attach in potential_reactions.items():
        mask = [True 
                if attach_label in d_potential_attach 
                else False
                for attach_label in fragment_attach_labels]
                
        valid_action_masks[attach_label_1] = torch.tensor(mask, dtype=torch.bool)
        
    return valid_action_masks


def select_mol_with_symbols(mols: list[Union[Mol, Fragment]],
                            z_list: list[int]):
    mol_is_included = []
    for mol in mols:
        if isinstance(mol, Fragment):
            frag = Fragment(mol=mol,
                            protections=mol.protections)
            frag.unprotect()
            mol = frag
        mol = Chem.RemoveHs(mol)
        included = True
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z not in z_list:
                included = False
                break
        mol_is_included.append(included)
        
    out_mols = [mol 
                for mol, included in zip(mols, mol_is_included)
                if included]
    
    return out_mols


def get_rotated_fragments(protected_fragments,
                            torsion_angles_deg: list[float]) -> dict[str, Fragment]:
        
    print('Rotate fragments')
    rotated_fragments = []
    for protected_fragment in tqdm(protected_fragments):
        for torsion_value in torsion_angles_deg:
            new_fragment = Fragment(protected_fragment,
                                    protections=protected_fragment.protections)
            rotation = Rotation.from_euler('x', torsion_value)
            rotate_conformer(new_fragment.GetConformer(), rotation)
            rotated_fragments.append(new_fragment)
            
    return rotated_fragments

def get_neighbor_symbol(fragment: Fragment):
    aps = fragment.get_attach_points()
    ap_atom_id = list(aps.keys())[0]
    ap_atom = fragment.GetAtomWithIdx(ap_atom_id)
    neighbors = ap_atom.GetNeighbors()
    assert len(neighbors) == 1
    return neighbors[0].GetSymbol()


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