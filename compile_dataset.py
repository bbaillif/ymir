import logging
import numpy as np
import pandas as pd

from rdkit import Chem
from tqdm import tqdm
from ymir.fragment_library import FragmentLibrary
from ymir.params import EMBED_HYDROGENS, TORSION_ANGLES_DEG
from ymir.atomic_num_table import AtomicNumberTable
from ymir.utils.fragment import (select_mol_with_symbols,
                                 get_masks,
                                 get_seeds,
                                 ConstructionSeed,
                                 center_fragments,
                                 get_rotated_fragments)
from ymir.molecule_builder import add_fragment_to_seed
from ymir.data import (Fragment,
                       Complex)
from ymir.metrics import VinaScorer, VinaScore
from ymir.bond_distance import MedianBondDistance
from ymir.utils.spatial import rotate_conformer, translate_conformer, Rotation
from scipy.spatial.distance import euclidean

median_bond_distance = MedianBondDistance()

logging.basicConfig(filename='compile_dataset.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode="w")

removeHs = not EMBED_HYDROGENS
fragment_library = FragmentLibrary(removeHs=removeHs)
ligands = fragment_library.ligands
protected_fragments = fragment_library.protected_fragments

# Filter out large ligands
ligands = [ligand 
           for ligand in ligands 
           if ligand.GetNumHeavyAtoms() < 50]

z_list = [0, 6, 7, 8, 16, 17]
if EMBED_HYDROGENS:
    z_list.append(1)
z_table = AtomicNumberTable(zs=z_list)
# Remove ligands having at least one heavy atom not in list
ligands = select_mol_with_symbols(ligands,
                                  z_list)

# Remove fragment having at least one heavy atom not in list
protected_fragments = select_mol_with_symbols(protected_fragments,
                                              z_list)

# Filter out fragments with more than 1 attachment point
n_attaches = []
for fragment in protected_fragments:
    frag_copy = Fragment(mol=fragment,
                         protections=fragment.protections)
    frag_copy.unprotect()
    attach_points = frag_copy.get_attach_points()
    n_attach = len(attach_points)
    n_attaches.append(n_attach)
    
protected_fragments = [fragment 
                       for fragment, n in zip(protected_fragments, n_attaches)
                       if n == 1]

protected_fragments_smiles = [Chem.MolToSmiles(frag) for frag in protected_fragments]

# TOREMOVE
ligands = ligands[:100]

# Get protein paths corresponding to ligands
protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)
    
assert len(ligands) == len(protein_paths)

# Get complexes with a correct Vina setup, and having a one-attach-point fragment in our list
not_working_protein = []
complexes: list[Complex] = []
seed_to_complex: list[int] = []
all_seeds: list[ConstructionSeed] = []
complex_counter = 0
for protein_path, ligand in tqdm(zip(protein_paths, ligands), total=len(protein_paths)):
    
    seeds = get_seeds(ligand)
    correct_seeds = []
    for seed in seeds:
        construct, removed_fragment = seed
        attach_points = construct.get_attach_points()
        assert len(attach_points) == 1
        if 7 in attach_points.values():
            continue
        
        frag_smiles = Chem.MolToSmiles(removed_fragment)
        if frag_smiles in protected_fragments_smiles:
            correct_seeds.append(seed)
    
    if len(correct_seeds) > 0:
        try:
            complx = Complex(ligand, protein_path)
            vina_scorer = VinaScorer(complx.vina_protein) # Generate the Vina protein file
            
        except Exception as e:
            logging.warning(f'Error on {protein_path}: {e}')
            not_working_protein.append(protein_path)
            
        else:
            complexes.append(complx)
            for seed in correct_seeds:
                all_seeds.append(seed)
                seed_to_complex.append(complex_counter)
            complex_counter += 1
        
center_fragments(protected_fragments)
rotated_fragments = get_rotated_fragments(protected_fragments, TORSION_ANGLES_DEG)

def get_neighbor_symbol(fragment: Fragment):
    aps = fragment.get_attach_points()
    ap_atom_id = list(aps.keys())[0]
    ap_atom = fragment.GetAtomWithIdx(ap_atom_id)
    neighbors = ap_atom.GetNeighbors()
    assert len(neighbors) == 1
    return neighbors[0].GetSymbol()

rotated_fragments_neighbors = [get_neighbor_symbol(fragment) 
                               for fragment in rotated_fragments]

valid_action_masks = get_masks(rotated_fragments)
        
def construct_to_frame(construct: Fragment) -> tuple[Rotation, np.ndarray]:
    for atom in construct.GetAtoms():
        if atom.GetAtomicNum() == 0:
            attach_point = atom
            break
    
    # Align the neighbor ---> attach point vector to the x axis: (0,0,0) ---> (1,0,0)
    # Then translate such that the neighbor is (0,0,0)
    neighbor = attach_point.GetNeighbors()[0]
    neighbor_id = neighbor.GetIdx()
    attach_id = attach_point.GetIdx()
    positions = construct.GetConformer().GetPositions()
    # neighbor_attach = positions[[neighbor_id, attach_id]]
    # distance = euclidean(neighbor_attach[0], neighbor_attach[1])
    # x_axis_vector = np.array([[0,0,0], [distance,0,0]])
    neighbor_pos = positions[neighbor_id]
    attach_pos = positions[attach_id]
    neighbor_attach = attach_pos - neighbor_pos
    distance = euclidean(attach_pos, neighbor_pos)
    x_axis_vector = np.array([distance, 0, 0])
    # import pdb;pdb.set_trace()
    rotation, rssd = Rotation.align_vectors(a=x_axis_vector.reshape(-1, 3), b=neighbor_attach.reshape(-1, 3))
    rotate_conformer(conformer=construct.GetConformer(),
                        rotation=rotation)
    # rotate_conformer(conformer=self.pocket_mol.GetConformer(),
    #                     rotation=rotation)
    
    positions = construct.GetConformer().GetPositions()
    neighbor_pos = positions[neighbor_id]
    translation = -neighbor_pos
    translate_conformer(conformer=construct.GetConformer(),
                        translation=translation)
    
    return (rotation, translation)
        
def reverse_transformations(ligand,
                            transformations):
    
    for transformation in reversed(transformations):
        if isinstance(transformation, Rotation):
            rotation_inv = transformation.inv()
            rotate_conformer(ligand.GetConformer(), rotation=rotation_inv)
        elif isinstance(transformation, np.ndarray):
            translation_inv = -transformation
            translate_conformer(ligand.GetConformer(), translation=translation_inv)
        else:
            import pdb;pdb.set_trace()
    
with Chem.SDWriter('/home/bb596/hdd/ymir/refined_rotated_fragments.sdf') as writer:
    for fragment in rotated_fragments:
        writer.write(fragment)
# with Chem.SDWriter('/home/bb596/hdd/ymir/constructs.sdf') as writer:
#     for fragment in :
#         writer.write(fragment)
    
n_torsions = len(TORSION_ANGLES_DEG)
n_fragments = 50
n_complexes = 100

# REMOVE LIMITS
try:
    frame_transformations = []
    rows = []
    for complx_i, complx in enumerate(tqdm(complexes[:n_complexes])):
        vina_scorer = VinaScorer(vina_protein=complx.vina_protein)
        vina_scorer.set_box_from_ligand(complx.ligand)
        vina_score = VinaScore(vina_scorer, minimized=True)
        for seed_i, complx_i_seed in enumerate(seed_to_complex):
            if complx_i_seed == complx_i:
                seed = all_seeds[seed_i]
                construct: Fragment = seed.construct
                construct_smiles = Chem.MolToSmiles(construct)
                transformations = construct_to_frame(construct)
                frame_transformations.append(transformations)
                construct_aps = construct.get_attach_points()
                assert len(construct_aps) == 1
                construct_label = list(construct_aps.values())[0]
                valid_action_mask = valid_action_masks[construct_label]
                added_fragments = [Fragment(fragment)
                                    for fragment, valid in zip(rotated_fragments, valid_action_mask)
                                    if valid]
                valid_fragment_ids = [idx 
                                      for idx, valid in enumerate(valid_action_mask)
                                      if valid]
                construct_neighbor_symbol = get_neighbor_symbol(construct)
                bond_tuples = []
                for fragment_i, fragment in enumerate(added_fragments[:n_fragments*n_torsions]):
                    fragment_neighbor_symbol = get_neighbor_symbol(fragment)
                    mbd = median_bond_distance.get_mbd(construct_neighbor_symbol, 
                                                    fragment_neighbor_symbol)
                    translation = [mbd, 0, 0]
                    translate_conformer(fragment.GetConformer(), translation)
                    product = add_fragment_to_seed(seed=construct,
                                                    fragment=fragment)
                    reverse_transformations(product, transformations)
                    Chem.SanitizeMol(product)
                    scores = vina_score.get([product], add_hydrogens=True)
                    import pdb;pdb.set_trace()
                    vina_scorer._vina.write_pose('optimized_pose.pdbqt')
                    
                    
                    score = scores[0]
                    
                    # fragment_smiles = Chem.MolToSmiles(fragment)
                    rotation_i = fragment_i % n_torsions
                    fragment_id = valid_fragment_ids[fragment_i]
                    
                    row = {'seed_i' : seed_i,
                        'protein_path' : complx.protein_path,
                        'construct_smiles': construct_smiles,
                        # 'fragment_smiles': fragment_smiles,
                        'fragment_id': fragment_id,
                        'rotation_i': rotation_i,
                        'vina_score': score}
                    rows.append(row)
                    
        # import pdb;pdb.set_trace()
                
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
except Exception as e:
    print(e)
    logging.warning(str(e))
    # import pdb;pdb.set_trace()
    
df = pd.DataFrame(rows)
df.to_csv('rows.csv')
# import pdb;pdb.set_trace()