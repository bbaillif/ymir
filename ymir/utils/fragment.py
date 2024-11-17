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
from ymir.molecule_builder import potential_reactions, add_fragment_to_seed
from typing import Union, NamedTuple
from tqdm import tqdm
from collections import Counter
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

def get_neighbor_id_for_atom_id(mol: Mol,
                                atom_id: int) -> int:
    atom = mol.GetAtomWithIdx(atom_id)
    neighbors = atom.GetNeighbors()
    assert len(neighbors) == 1
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
    
    
def get_no_attach_mapping(atom_idxs: list[int],
                          fragment: Fragment):
    attach_points = fragment.get_attach_points(include_protected=True).keys()
    new_mapping = [atom_idx 
                   for i, atom_idx in enumerate(atom_idxs) 
                   if i not in attach_points]
    return new_mapping
    
    
# def find_broken_bond(ligand: Mol, 
#                      seed: Fragment, 
#                      seed_i: int, 
#                      fragment: Fragment, 
#                      fragment_i: int, 
#                      frags_mol_atom_mapping: list[list[int]]):
#     try:
#         # if Chem.MolToSmiles(seed.mol) == '[3*]O[3*]' and Chem.MolToSmiles(fragment.mol) == '[1*]C([1*])=O':
#         #     import pdb;pdb.set_trace()
#         for attach_point, label in seed.get_attach_points().items():
#             neigh_id1 = get_neighbor_id_for_atom_id(seed.mol, attach_point)
#             mapping1 = frags_mol_atom_mapping[seed_i]
#             native_neigh_id1 = mapping1[neigh_id1]
#             # native_ap1 = mapping1[attach_point]
#             for attach_point2, label2 in fragment.get_attach_points().items():
#                 neigh_id2 = get_neighbor_id_for_atom_id(fragment.mol, attach_point2)
#                 mapping2 = frags_mol_atom_mapping[fragment_i]
#                 native_neigh_id2 = mapping2[neigh_id2]
#                 # native_ap2 = mapping2[attach_point2]

#                 potential_reactions_seed = potential_reactions[label]
#                 if label2 in potential_reactions_seed:
#                     bond = ligand.GetBondBetweenAtoms(native_neigh_id1, native_neigh_id2)
#                     if bond is not None:
#                         ap1_position = seed.mol.GetConformer().GetPositions()[attach_point]
#                         neigh2_position = fragment.mol.GetConformer().GetPositions()[neigh_id2]
#                         cond1 = ap1_position.tolist() == neigh2_position.tolist()
#                         ap2_position = fragment.mol.GetConformer().GetPositions()[attach_point2]
#                         neigh1_position = seed.mol.GetConformer().GetPositions()[neigh_id1]
#                         cond2 = ap2_position.tolist() == neigh1_position.tolist()
#                         if cond1 and cond2:
#                             return (neigh_id1, attach_point, neigh_id2, attach_point2)
                                
#     except Exception as e:
#         print(str(e))
#         import pdb;pdb.set_trace()
#     return None


def find_broken_bond(ligand: Mol, 
                     seed: Fragment, 
                     fragment: Fragment, 
                     seed_mapping: list[list[int]],
                     fragment_mapping: list[list[int]]):
    try:
        # if Chem.MolToSmiles(seed.mol) == '[3*]O[3*]' and Chem.MolToSmiles(fragment.mol) == '[1*]C([1*])=O':
        #     import pdb;pdb.set_trace()
        for attach_point, label in seed.get_attach_points().items():
            broken_bond = find_broken_bond_for_attach(attach_point, 
                                                      label, 
                                                      ligand, 
                                                      seed, 
                                                      fragment, 
                                                      seed_mapping, 
                                                      fragment_mapping)
            if broken_bond is not None:
                return broken_bond
            
                                
    except Exception as e:
        print(str(e))
        import pdb;pdb.set_trace()
    return None
    
    
def find_broken_bond_for_attach(attach_point: int,
                                label: int,
                                ligand: Mol, 
                                seed: Fragment, 
                                fragment: Fragment, 
                                seed_mapping: list[list[int]],
                                fragment_mapping: list[list[int]]):
    
    neigh_id1 = get_neighbor_id_for_atom_id(seed.mol, attach_point)
    native_neigh_id1 = seed_mapping[neigh_id1]
    # native_ap1 = mapping1[attach_point]
    for attach_point2, label2 in fragment.get_attach_points().items():
        neigh_id2 = get_neighbor_id_for_atom_id(fragment.mol, attach_point2)
        native_neigh_id2 = fragment_mapping[neigh_id2]
        # native_ap2 = mapping2[attach_point2]

        potential_reactions_seed = potential_reactions[label]
        if label2 in potential_reactions_seed:
            bond = ligand.GetBondBetweenAtoms(native_neigh_id1, native_neigh_id2)
            if bond is not None:
                ap1_position = seed.mol.GetConformer().GetPositions()[attach_point]
                neigh2_position = fragment.mol.GetConformer().GetPositions()[neigh_id2]
                cond1 = ap1_position.tolist() == neigh2_position.tolist()
                ap2_position = fragment.mol.GetConformer().GetPositions()[attach_point2]
                neigh1_position = seed.mol.GetConformer().GetPositions()[neigh_id1]
                cond2 = ap2_position.tolist() == neigh1_position.tolist()
                if cond1 and cond2:
                    return (neigh_id1, attach_point, neigh_id2, attach_point2)
                
    return None
    
def get_fragments_from_mol(mol: Mol) -> list[Fragment]:
    
    # fragments = Chem.BRICS.BRICSDecompose(mol,
    #                                     minFragmentSize=2, 
    #                                     keepNonLeafNodes=False, 
    #                                     returnMols=True)
    
    # combined_mol = fragments[0]
    # for fragment in fragments[1:]:
    #     combined_mol = Chem.CombineMols(combined_mol, fragment)
    
    # import pdb;pdb.set_trace()
    
    pieces = Chem.BRICS.BreakBRICSBonds(mol)
    frags_mol_atom_mapping = []
    fragments: list[Mol] = Chem.GetMolFrags(pieces, asMols=True, fragsMolAtomMapping=frags_mol_atom_mapping)

    for frag in fragments:
        Chem.AssignStereochemistryFrom3D(frag)
        
    fragments: list[Fragment] = [Fragment(frag) for frag in fragments]

    # no_attach_mappings = []
    # for mapping, fragment in zip(frags_mol_atom_mapping, fragments):
    #     no_attach_mappings.append(get_no_attach_mapping(mapping, fragment))
        
    # frags_mol_atom_mapping = no_attach_mappings

    # Re-attach fragments with only one atom
    # bonds_to_link = {}
    # new_frags = []
    # merged_frag_idxs = []
    # new_mappings = []
    
    frag_i = 0
    while frag_i < len(fragments):
        frag = fragments[frag_i]
        if frag.mol.GetNumHeavyAtoms() == 1 and len(frag.get_attach_points()) == 1:
            for frag_i2 in range(len(fragments)):
                if frag_i != frag_i2:
                    frag2 = fragments[frag_i2]
                    mapping1 = frags_mol_atom_mapping[frag_i]
                    mapping2 = frags_mol_atom_mapping[frag_i2]
                    broken_bond = find_broken_bond(mol, frag, frag2,
                                                   mapping1, mapping2)
                    if broken_bond is not None:
                        # if Chem.MolToSmiles(frag.mol) == '[3*]O[H]':
                        #     import pdb;pdb.set_trace()
                        neigh1_id, ap1_id, neigh2_id, ap2_id = broken_bond
                        # if frag_i > frag_i2:
                        #     frag_i, frag_i2 = frag_i2, frag_i
                        #     ap1_id, ap2_id = ap2_id, ap1_id
                        # frag1 = fragments[frag_i]
                        # frag2 = fragments[frag_i2]
                        frag.protect(atom_ids_to_keep=[ap1_id])
                        frag2.protect(atom_ids_to_keep=[ap2_id])
                        
                        product, seed_to_product_mapping, fragment_to_product_mapping = add_fragment_to_seed(frag, frag2)
                        # seed_to_product_mapping = get_no_attach_mapping(seed_to_product_mapping, frag)
                        # fragment_to_product_mapping = get_no_attach_mapping(fragment_to_product_mapping, frag2)
                        
                        new_mapping = merge_mappings(mapping1,
                                                     mapping2,
                                                     seed_to_product_mapping,
                                                     fragment_to_product_mapping,
                                                     ap1_id,
                                                     ap2_id)
                        
                        # new_mapping1 = [mapping1[]seed_to_product_mapping[i] for i in mapping1]
                        # mapping2 = no_attach_mappings[frag_i2]
                        # new_mapping2 = [fragment_to_product_mapping[i] for i in mapping2]
                        # new_mapping = new_mapping1 + new_mapping2
                        if frag_i < frag_i2:
                            frag_i, frag_i2 = frag_i2, frag_i
                        fragments[frag_i] = product
                        frags_mol_atom_mapping[frag_i] = new_mapping
                        
                        fragments.pop(frag_i2)
                        frags_mol_atom_mapping.pop(frag_i2)
                        
                        if frag_i > frag_i2:
                            frag_i -= 1 # because we removed the small fragment that was on the right side
                        break
                        
        frag_i += 1
    
    # for frag_i, frag in enumerate(frags):
    #     if frag.mol.GetNumHeavyAtoms() == 1:
    #         for frag_i2, frag2 in enumerate(frags):
    #             if frag_i != frag_i2:
    #                 broken_bond = find_broken_bond(mol, 
    #                                                frag, frag_i, 
    #                                                frag2, frag_i2, 
    #                                                frags_mol_atom_mapping)
    #                 if broken_bond is not None:
    #                     neigh1_id, ap1_id, neigh2_id, ap2_id = broken_bond
    #                     bond_to_link = (ap1_id, frag_i2, ap2_id)
    #                     bonds_to_link[frag_i] = bond_to_link
    #                     merged_frag_idxs.append(frag_i2)
    #                     break
    #     else:
    #         if frag_i not in merged_frag_idxs:
    #             new_frags.append(frag)
    #             new_mappings.append(frags_mol_atom_mapping[frag_i])
    
    # for frag_i, bond in bonds_to_link.items():
    #     ap1_id, frag_i2, ap2_id = bond
    #     frag1 = frags[frag_i]
    #     frag2 = frags[frag_i2]
    #     frag1.protect(atom_ids_to_keep=[ap1_id])
    #     frag2.protect(atom_ids_to_keep=[ap2_id])
    #     new_frag = add_fragment_to_seed(frag1, frag2)
    #     new_frags.append(new_frag)
    #     new_mapping = no_attach_mappings[frag_i] + no_attach_mappings[frag_i2]
    #     new_mappings.append(new_mapping)
    
    # fragments = new_frags
    # frags_mol_atom_mapping = new_mappings
    
    # get canon order of canon fragments
    min_idxs = []
    for mapping in frags_mol_atom_mapping:
        min_idx = min(mapping)
        min_idxs.append(min_idx)
        
    sort_idx = np.argsort(min_idxs)
    final_fragments = []
    final_mappings = []
    for fragment_i in sort_idx:
        current_fragment = fragments[fragment_i]
        current_mapping = frags_mol_atom_mapping[fragment_i]
        final_fragment, final_mapping = get_canon_fragment(current_fragment, 
                                                           current_mapping)
        final_fragments.append(final_fragment)
        final_mappings.append(final_mapping)
    
    return final_fragments, final_mappings


def merge_mappings(mapping1: list[int],
                   mapping2: list[int],
                   seed_to_product_mapping: list[int],
                   fragment_to_product_mapping: list[int],
                   ap_seed: int,
                   ap_fragment: int) -> list[int]:
    new_mapping_size = len(mapping1) + len(mapping2) - 2
    new_mapping = [None] * new_mapping_size
                        
    try:
        for seed_i, product_i in enumerate(seed_to_product_mapping):
            if seed_i != ap_seed:
                new_mapping[product_i] = mapping1[seed_i]
        for seed_i, product_i in enumerate(fragment_to_product_mapping):
            if seed_i != ap_fragment:
                new_mapping[product_i] = mapping2[seed_i]
    except:
        import pdb;pdb.set_trace()
        
    if not all([m is not None for m in new_mapping]):
        import pdb;pdb.set_trace()
        
    return new_mapping


def get_canon_fragment(frag: Fragment,
                       frags_mol_atom_mapping: list[int]) -> Fragment:
    # assert len(frag.get_attach_points()) == 1
    # frag_h = Chem.AddHs(frag.mol, addCoords=True)
    # assert len(frag_h.GetAtoms()) == len(frag.mol.GetAtoms())
    smiles = Chem.MolToSmiles(frag.mol)
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    canon_mol = Chem.MolFromSmiles(smiles, ps)
    # canon_mol = Chem.AddHs(canon_mol)
    one2two = canon_mol.GetSubstructMatch(frag.mol)
    if len(one2two) != frag.mol.GetNumAtoms():
        import pdb;pdb.set_trace()
    mol_copy = Mol(frag.mol)
    canon_mol.AddConformer(mol_copy.GetConformer())
    for atom_id1, atom_id2 in enumerate(one2two):
        point3d = frag.mol.GetConformer().GetAtomPosition(atom_id1)
        try:
            canon_mol.GetConformer().SetAtomPosition(atom_id2, point3d)
        except:
            import pdb;pdb.set_trace()
    canon_frags_mol_atom_mapping = [None] * canon_mol.GetNumAtoms()
    for one_i, two_i in enumerate(one2two):
        canon_frags_mol_atom_mapping[two_i] = frags_mol_atom_mapping[one_i]
    canon_frag = Fragment(canon_mol)
    # if smiles != Chem.MolToSmiles(canon_frag.mol):
    #     import pdb;pdb.set_trace()
    return canon_frag, canon_frags_mol_atom_mapping


def get_unique_fragments_from_mols(mols: list[Mol]) -> list[Fragment]:
    
    smiles_fragments = {}
    smiles_counter = Counter()
    for mol in tqdm(mols):
        fragments, frags_mol_atom_mapping = get_fragments_from_mol(mol)
        if len(fragments) > 1:
            for i, frag in enumerate(fragments):
                attach_points = frag.get_attach_points()
                frag_mol = frag.mol
                if len(attach_points) > 0:
                    Chem.SanitizeMol(frag_mol)
                    smiles = Chem.MolToSmiles(frag_mol, 
                                            # allHsExplicit=True
                                            )
                    new_name = mol.GetProp('_Name') + f'_{i}'
                    frag_mol.SetProp('_Name', 
                                new_name)
                    # if not smiles in smiles_counter:
                    smiles_counter.update([smiles])
                    if not smiles in smiles_fragments:
                        smiles_fragments[smiles] = frag
                    
    ranked_list = smiles_counter.most_common()
    unique_fragments = [smiles_fragments[smiles] for smiles, _ in ranked_list]
                
    return unique_fragments


class ConstructionSeed(NamedTuple): 
    ligand: Mol
    removed_fragment_atom_idxs: list[int]
    
    
    def decompose(self):
        brics_bonds = Chem.BRICS.FindBRICSBonds(self.ligand)
        broken = False
        for bond in brics_bonds:
            t_bond, t_labels = bond
            if (t_bond[0] in self.removed_fragment_atom_idxs) or (t_bond[1] in self.removed_fragment_atom_idxs):
                new_mol = Chem.BRICS.BreakBRICSBonds(self.ligand, bonds=[bond])
                broken = True
                break
        if not broken:
            import pdb;pdb.set_trace()
        
        frags_mol_atom_mapping = []
        new_frags = Chem.GetMolFrags(new_mol, asMols=True, fragsMolAtomMapping=frags_mol_atom_mapping)
        frags_mol_atom_mapping = [get_no_attach_mapping(mapping, self.ligand) for mapping in frags_mol_atom_mapping]
        assert len(new_frags) == 2
        frag1, frag2 = new_frags
        if self.removed_fragment_atom_idxs == frags_mol_atom_mapping[0]:
            construct = Fragment(frag2)
            removed_fragment = Fragment(frag1)
        elif self.removed_fragment_atom_idxs == frags_mol_atom_mapping[1]:
            construct = Fragment(frag1)
            removed_fragment = Fragment(frag2)
        else:
            import pdb;pdb.set_trace()
            
        return construct, removed_fragment, bond


def get_seeds(ligand: Mol) -> list[ConstructionSeed]:
    fragments, frags_mol_atom_mapping = get_fragments_from_mol(ligand)
    
    fragments_1ap: list[Fragment] = []
    fragments_idxs: list[int] = []
    for frag_i, fragment in enumerate(fragments):
        aps = fragment.get_attach_points()
        if len(aps) == 1:
            fragments_1ap.append(fragment)
            fragments_idxs.append(frag_i)
            
    seeds = []
    for frag_i in fragments_idxs :
        removed_fragment_atom_idxs = frags_mol_atom_mapping[frag_i]
        construction_seed = ConstructionSeed(ligand,
                                             removed_fragment_atom_idxs)
        seeds.append(construction_seed)
            
    return seeds


# Align the attach point ---> neighbor vector to the x axis: (0,0,0) ---> (1,0,0)
# Then translate such that the neighbor is (0,0,0)
def center_fragments(protected_fragments: list[Fragment],
                     attach_to_neighbor: bool = True,
                     neighbor_is_zero: bool = True,
                     ) -> list[list[Union[np.ndarray, Rotation]]]:
    all_transformations = []
    for fragment in protected_fragments:
        transformations = center_fragment(fragment,
                                          attach_to_neighbor,
                                          neighbor_is_zero)
        all_transformations.append(transformations)
        
    return all_transformations
        
       
# Align the attach point ---> neighbor vector to the x axis: (0,0,0) ---> (1,0,0), if attach_to_neighbor
# Else, neighbor ---> attach is the x axis.
# Then translate such that the neighbor is (0,0,0) if neighbor_is_zero
# Else the attach is the centre
def center_fragment(fragment: Fragment,
                    attach_to_neighbor: bool = True,
                    neighbor_is_zero: bool = True) -> list[Union[np.ndarray, Rotation]]:
    assert len(fragment.get_attach_points()) == 1
    mol = fragment.mol
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            attach_point = atom
            break
    neighbor = attach_point.GetNeighbors()[0]
    neighbor_id = neighbor.GetIdx()
    attach_id = attach_point.GetIdx()
    positions = mol.GetConformer().GetPositions()
    # neighbor_attach = positions[[neighbor_id, attach_id]]
    # distance = euclidean(neighbor_attach[0], neighbor_attach[1])
    # x_axis_vector = np.array([[0,0,0], [distance,0,0]])
    neighbor_pos = positions[neighbor_id]
    attach_pos = positions[attach_id]
    if attach_to_neighbor:
        direction = neighbor_pos - attach_pos
    else:
        direction = attach_pos - neighbor_pos
    distance = euclidean(neighbor_pos, attach_pos)
    x_axis_vector = np.array([distance, 0, 0])
    # import pdb;pdb.set_trace()
    # rotation, rssd = Rotation.align_vectors(a=x_axis_vector.reshape(-1, 3), b=direction.reshape(-1, 3))
    rotation, rssd = Rotation.align_vectors(a=x_axis_vector, b=direction)
    rotate_conformer(conformer=mol.GetConformer(),
                        rotation=rotation)
    
    positions = mol.GetConformer().GetPositions()
    if neighbor_is_zero:
        neighbor_pos = positions[neighbor_id]
        translation = -neighbor_pos
    else:
        attach_pos = positions[attach_id]
        translation = -attach_pos
    translate_conformer(conformer=mol.GetConformer(),
                        translation=translation)
    
    transformations = [rotation, translation]
    return transformations


def get_masks(attach_labels: list[list[int]]):

    valid_action_masks: dict[int, list[bool]] = {}
    for attach_label_1, d_potential_attach in potential_reactions.items():
        mask = []
        for attach_labels_2 in attach_labels:
            mask.append(any([attach_label_2 in d_potential_attach 
                             for attach_label_2 in attach_labels_2]))
                
        valid_action_masks[attach_label_1] = torch.tensor(mask, dtype=torch.bool)
        
    return valid_action_masks


def select_mol_with_symbols(mols: list[Union[Mol, Fragment]],
                            z_list: list[int]) -> list[Union[Mol, Fragment]]:
    mol_is_included = []
    for mol in mols:
        if isinstance(mol, Fragment):
            frag = Fragment.from_fragment(mol)
            frag.unprotect()
            mol = frag.mol
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
    all_rotated_fragments = []
    for protected_fragment in protected_fragments:
        rotated_fragments = []
        for torsion_value in torsion_angles_deg:
            new_fragment = Fragment.from_fragment(protected_fragment)
            rotation = Rotation.from_euler('x', torsion_value, degrees=True)
            rotate_conformer(new_fragment.mol.GetConformer(), rotation)
            rotated_fragments.append(new_fragment)
        all_rotated_fragments.append(rotated_fragments)
            
    return all_rotated_fragments

def get_neighbor_symbol(fragment: Fragment):
    aps = fragment.get_attach_points()
    ap_atom_id = list(aps.keys())[0]
    ap_atom = fragment.mol.GetAtomWithIdx(ap_atom_id)
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


def find_mappings(mcs, mol1, mol2):
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    Chem.SanitizeMol(mcs_mol)
    matches1 = mol1.GetSubstructMatches(mcs_mol, uniquify=False)
    matches2 = mol2.GetSubstructMatches(mcs_mol, uniquify=False)
    # import pdb;pdb.set_trace()
    mappings = []
    for match1 in matches1:
        for match2 in matches2:
            one2mcs = {res: idx for idx, res in enumerate(match1)}
            mcs2two = {idx: res for idx, res in enumerate(match2)}
            one2two = {res: mcs2two[idx] for res, idx in one2mcs.items()}
            mappings.append(one2two)
    return mappings
    
    # assert len(match) == mol1.GetNumHeavyAtoms()
    # # Try using bond order assignement
    # standard_mol = AssignBondOrdersFromTemplate(mol1, standard_mol)
    # mcs = FindMCS([mol1, mol2])
    # mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    # match = standard_mol.GetSubstructMatch(mcs_mol)
    # if not len(match) == mol1.GetNumAtoms() :
    #     import pdb;pdb.set_trace()
    #     raise Exception('No match found between template and actual mol')
    
    # self_match = mol1.GetSubstructMatch(mcs_mol)
    # assert len(match) == len(self_match)
    # self2mcs = {res: idx for idx, res in enumerate(self_match)}
    # self2new = {idx: match[res] for idx, res in self2mcs.items()}
    # new_match = []
    # for i in range(len(self2new)):
    #     new_match.append(self2new[i])
    # mapping = [(mol1i, mol2i) for mol1i, mol2i in enumerate(new_match)]
    # return mapping