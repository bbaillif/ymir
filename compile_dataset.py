import time
import torch
import random
import numpy as np
import logging
import pickle
import os
# import wandb

from torch import nn
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Mol
from typing import Union
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch_geometric.data import Batch, Data
from e3nn import o3

from ymir.fragment_library import FragmentLibrary
from ymir.data.structure import VinaProtein, Pocket, Protein
from ymir.utils.fragment import (get_seeds, 
                                 center_fragments, 
                                 center_fragment,
                                 get_masks,
                                 select_mol_with_symbols,
                                 get_neighbor_symbol,
                                 ConstructionSeed)
from ymir.utils.spatial import translate_conformer, reverse_transformations

from ymir.atomic_num_table import AtomicNumberTable

from ymir.env import (FragmentBuilderEnv, 
                            BatchEnv)
from ymir.policy import Agent, Action
from ymir.data import Fragment
from ymir.params import (EMBED_HYDROGENS, 
                         HIDDEN_IRREPS,
                         SEED,
                         VINA_DATASET_PATH)
from ymir.metrics.activity import VinaScore, VinaScorer
from ymir.metrics.activity.vina_cli import VinaCLI
from ymir.bond_distance import MedianBondDistance
from ymir.molecule_builder import add_fragment_to_seed
from scipy.spatial.transform import Rotation

logging.basicConfig(filename='compile_dataset.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode="w")

seed = SEED
random.seed(seed)
np.random.seed(seed)

# n_complexes = 5

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_v1_{timestamp}"

removeHs = not EMBED_HYDROGENS
fragment_library = FragmentLibrary(removeHs=removeHs)
# ligands = fragment_library.ligands
# protected_fragments = fragment_library.protected_fragments

z_list = [0, 6, 7, 8, 16, 17]
if EMBED_HYDROGENS:
    z_list.append(1)
z_table = AtomicNumberTable(zs=z_list)

# Remove ligands having at least one heavy atom not in list
ligands = fragment_library.get_restricted_ligands(z_list)

# Remove fragment having at least one heavy atom not in list
protected_fragments = fragment_library.get_restricted_fragments(z_list)
           
protected_fragments_smiles = [Chem.MolToSmiles(frag) for frag in protected_fragments]

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)
    
assert len(ligands) == len(protein_paths)

logging.info('Loading complexes')

# Get complexes with a correct Vina setup, and having a one-attach-point fragment in our list
not_working_protein = []

# complexes: list[Complex] = []
correct_pdbqt_paths: list[str] = []
all_native_ligands: list[Mol] = []
seed_to_complex: list[int] = []
all_seeds: list[ConstructionSeed] = []
complex_counter = 0
for protein_path, ligand in tqdm(zip(protein_paths, ligands), total=len(protein_paths)):
    
    seeds = get_seeds(ligand)
    correct_seeds = []
    for seed in seeds:
        construct, removed_fragment, bond = seed.decompose()
        
        attach_points = construct.get_attach_points()
        assert len(attach_points) == 1
        if 7 in attach_points.values():
            continue
        
        frag_smiles = Chem.MolToSmiles(removed_fragment)
        if frag_smiles in protected_fragments_smiles:
            correct_seeds.append(seed)
    
    if len(correct_seeds) > 0:
        try:
            vina_protein = VinaProtein(pdb_filepath=protein_path)
            pdbqt_filepath = vina_protein.pdbqt_filepath
            pocket = Pocket(protein=Protein(pdb_filepath=vina_protein.protein_clean_filepath),
                            native_ligand=ligand) # Detect the short pocket situations
            
        except Exception as e:
            logging.warning(f'Error on {protein_path}: {e}')
            not_working_protein.append(protein_path)
            
        else:
            # complexes.append(complx)
            correct_pdbqt_paths.append(pdbqt_filepath)
            all_native_ligands.append(ligand)
            for seed in correct_seeds:
                all_seeds.append(seed)
                seed_to_complex.append(complex_counter)
            complex_counter += 1
            
            # TO REMOVE IN FINAL
            # if complex_counter == n_complexes:
            #     break
         
dataset_path = '/home/bb596/hdd/ymir/dataset/'
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
         
vina_cli = VinaCLI()
native_ligands_h = [Chem.AddHs(mol, addCoords=True) 
                    for mol in all_native_ligands]
native_scores = vina_cli.get(receptor_paths=correct_pdbqt_paths,
                    native_ligands=all_native_ligands,
                    ligands=native_ligands_h)
            
n_seeds = len(all_seeds)

center_fragments(protected_fragments)
final_fragments = protected_fragments

valid_action_masks = get_masks(final_fragments)

logging.info(f'There are {len(final_fragments)} fragments')

mbd = MedianBondDistance()

# import pdb;pdb.set_trace()

rows = []
for seed_i, seed in enumerate(tqdm(all_seeds)):
    
    data_filepath = os.path.join(dataset_path, f'{seed_i}.p')
    # if not os.path.exists(data_filepath):
    
    construct, removed_fragment, bond = seed.decompose()
    complx_i = seed_to_complex[seed_i]
    protein_path = correct_pdbqt_paths[complx_i]
    
    # import pdb;pdb.set_trace()
    
    native_ligand = all_native_ligands[complx_i]
    native_score = native_scores[complx_i]

    transformations = center_fragment(construct,
                                        attach_to_neighbor=False)
    
    attach_points = construct.get_attach_points()
    assert len(attach_points) == 1
    attach_label = list(attach_points.values())[0]
    
    valid_action_mask = valid_action_masks[attach_label]
    # valid_action_idxs = [i for i, value in valid_action_mask if value]
    fragments = [final_fragments[i] for i, value in enumerate(valid_action_mask) if value]
        
    construct_neighbor_symbol = get_neighbor_symbol(construct)
    products = []
    for fragment in fragments:
        new_construct = Fragment(construct, construct.protections)
        new_fragment = Fragment(fragment, fragment.protections)
        fragment_neighbor_symbol = get_neighbor_symbol(fragment)
        distance = mbd.get_mbd(construct_neighbor_symbol, fragment_neighbor_symbol)
        translation = np.array([distance, 0, 0])
        translate_conformer(new_fragment.GetConformer(), translation)
        product = add_fragment_to_seed(seed=new_construct,
                                        fragment=new_fragment)
        products.append(product)
        
    ligands = []
    for product in products:
        reverse_transformations(product.GetConformer(), transformations)
        product.protect()
        Chem.SanitizeMol(product)
        mol = Chem.RemoveHs(product)
        mol_h = Chem.AddHs(mol, addCoords=True)
        ligands.append(mol_h)
    
    n_products = len(products)

    vina_cli = VinaCLI()
    receptor_paths = [protein_path
                    for _ in range(n_products)]
    native_ligands = [native_ligand
                    for _ in range(n_products)]
    scores = vina_cli.get(receptor_paths=receptor_paths,
                        native_ligands=native_ligands,
                        ligands=ligands)
    # relative_scores = [score - native_score for score in scores]
    
    row = {'protein_path' : protein_path,
        'ligand': native_ligand,
        'removed_fragment_atom_idxs': seed.removed_fragment_atom_idxs,
        'absolute_scores': scores,
        'native_score': native_score}
    # rows.append(row)
    
    with open(data_filepath, 'wb') as f:
        pickle.dump(row, f)
    
    
# with open(VINA_DATASET_PATH, 'wb') as f:
#     pickle.dump(rows, f)
    
import pdb;pdb.set_trace()