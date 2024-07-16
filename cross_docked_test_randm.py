import numpy as np
import torch
import os
import pickle

from tqdm import tqdm
from rdkit import Chem
from ymir.data.cross_docked import CrossDocked
from ymir.data.structure import Complex
from ymir.utils.fragment import get_fragments_from_mol, get_rotated_fragments, get_masks, center_fragments
from ymir.metrics.activity.smina_cli import SminaCLI
from ymir.env import FragmentBuilderEnv, BatchEnv
from ymir.fragment_library import FragmentLibrary
from ymir.atomic_num_table import AtomicNumberTable
from ymir.params import EMBED_HYDROGENS, TORSION_ANGLES_DEG, SCORING_FUNCTION
from ymir.data import Fragment
from torch_geometric.data import Batch
from ymir.policy import Agent, CategoricalMasked, Action
from ymir.params import (EMBED_HYDROGENS, 
                         SEED,
                         POCKET_RADIUS,
                         TORSION_ANGLES_DEG,
                         SCORING_FUNCTION,
                         CROSSDOCKED_POCKET10_PATH)

n_gen_molecules = 100
batch_size = 25
n_max_steps = 10
pocket_feature_type = 'graph'
n_fragments = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_list = [0, 6, 7, 8, 16, 17]
if EMBED_HYDROGENS:
    z_list.append(1)
z_table = AtomicNumberTable(zs=z_list)

cross_docked_train = CrossDocked()
cd_ligands = cross_docked_train.get_ligands()
cd_smiles = []
unique_ligands = []
cd_is = []
for cd_i, ligand in enumerate(cd_ligands):
    smiles = Chem.MolToSmiles(ligand)
    if smiles not in cd_smiles:
        cd_smiles.append(smiles)
        unique_ligands.append(ligand)
        cd_is.append(cd_i)

fragment_library = FragmentLibrary(ligands=unique_ligands,
                                   removeHs=False,
                                   subset='cross_docked')

restricted_fragments = fragment_library.get_restricted_fragments(z_list, 
                                                                max_attach=4, 
                                                                max_torsions=2,
                                                                n_fragments=n_fragments,
                                                                get_unique=True
                                                                )

synthon_smiles: list[str] = []
protected_fragments: list[Fragment] = []
attach_labels: list[list[int]] = []
for smiles, t in restricted_fragments.items():
    fragment, labels = t
    synthon_smiles.append(smiles)
    protected_fragments.append(fragment)
    attach_labels.append(labels)

if SCORING_FUNCTION == 'smina':
    for pfrag in protected_fragments:
        assert all([pfrag.mol.GetAtomWithIdx(atom_id).GetSymbol() != 'H' 
                    for atom_id in pfrag.protections.keys()])
        # pfrag.mol.Debug()
        pfrag.remove_hs()

center_fragments(protected_fragments)

torsion_angles_deg = TORSION_ANGLES_DEG
n_torsions = len(torsion_angles_deg)
rotated_fragments = get_rotated_fragments(protected_fragments, torsion_angles_deg)

valid_action_masks = get_masks(attach_labels)

cross_docked_test = CrossDocked(subset='test')
cd_split = cross_docked_test.get_split()
cd_ligands = cross_docked_test.get_ligands()

def get_seeds_complx(params) -> tuple[list[Fragment], Complex]:
    protein_path, pocket_path, ligand = params
    fragments, frags_mol_atom_mapping = get_fragments_from_mol(ligand)
        
    current_seeds = []
    complx = None
    pocket = None
    if len(fragments) > 1:
        
        if not all([7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()] 
                    for fragment in fragments]):
        
            try:
                pocket_full_path = os.path.join(CROSSDOCKED_POCKET10_PATH, pocket_path)
                pocket = Chem.MolFromPDBFile(pocket_full_path, sanitize=False)
                if pocket.GetNumAtoms() > 20 :
                    complx = Complex(ligand, protein_path, pocket_full_path)
                    # complx._pocket = Chem.MolFromPDBFile(pocket_full_path)
                    # if all([atom.GetAtomicNum() in z_list for atom in complx.pocket.mol.GetAtoms()]):
                    pf = complx.vina_protein.pdbqt_filepath
            
                    for fragment in fragments:
                        if not any([7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()]]):
                            current_seed = Fragment.from_fragment(fragment)
                            current_seeds.append(current_seed)
            except:
                print(f'No pdbqt file for {protein_path}')
                
    return current_seeds, complx, pocket

protein_paths = []
considered_ligands = []
pocket_paths = []
ligand_subpaths = []
for duo, ligand in tqdm(zip(cd_split, cd_ligands), total=len(cd_split)):
    if all([atom.GetAtomicNum() in z_list for atom in ligand.GetAtoms()]):
        pocket_path, ligand_subpath = duo
        protein_paths.append(cross_docked_test.get_original_structure_path(ligand_filename=ligand_subpath))
        considered_ligands.append(ligand)
        pocket_paths.append(pocket_path)
        ligand_subpaths.append(ligand_subpath)

complexes: list[Complex] = []
seeds: list[Fragment] = []
seed2complex = []
generation_sequences = []
selected_ligand_filenames = []
z = zip(protein_paths, pocket_paths, considered_ligands, ligand_subpaths)
for protein_path, pocket_path, ligand, ligand_subpath in tqdm(z, 
                                                            total=len(protein_paths)):
    current_seeds, complx, pocket = get_seeds_complx((protein_path, pocket_path, ligand))
    if len(current_seeds) > 0:
        complexes.append(complx)
        seeds.extend(current_seeds)
        seed2complex.extend([len(complexes) - 1 for _ in current_seeds])
        generation_sequences.extend([None for _ in current_seeds])
        selected_ligand_filenames.append(ligand_subpath)

n_complexes = len(complexes)

initial_scores_path = f'/home/bb596/hdd/ymir/initial_scores_{n_complexes}cplx_100frags_{n_torsions}rots_{SCORING_FUNCTION}_min_cd_test.pkl'
if not os.path.exists(initial_scores_path):
    receptor_paths = [complexes[seed2complex[seed_i]].vina_protein.pdbqt_filepath for seed_i in range(len(seeds))]
    seed_copies = [Fragment.from_fragment(seed) for seed in seeds]
    for seed_copy in seed_copies:
        seed_copy.protect()
    mols = [seed_copy.mol for seed_copy in seed_copies]
    smina_cli = SminaCLI()
    initial_scores = smina_cli.get(receptor_paths=receptor_paths,
                                ligands=mols)
    assert len(initial_scores) == len(seeds)
    
    with open(initial_scores_path, 'wb') as f:
        pickle.dump(initial_scores, f)
        
else:
    with open(initial_scores_path, 'rb') as f:
        initial_scores = pickle.load(f)

native_scores_path = f'/home/bb596/hdd/ymir/native_scores_{n_complexes}cplx_100frags_{n_torsions}rots_{SCORING_FUNCTION}_min_cd_test.pkl'
if not os.path.exists(native_scores_path):
    receptor_paths = [complx.vina_protein.pdbqt_filepath for complx in complexes]
    smina_cli = SminaCLI(score_only=False)
    scores = smina_cli.get(receptor_paths=receptor_paths,
                        ligands=[complx.ligand for complx in complexes])
    assert len(scores) == len(complexes)
    native_scores = [scores[i] for i in seed2complex]
    assert len(native_scores) == len(seeds)
        
    with open(native_scores_path, 'wb') as f:
        pickle.dump(native_scores, f)
    
else: 
    with open(native_scores_path, 'rb') as f:
        native_scores = pickle.load(f)


envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(rotated_fragments=rotated_fragments,
                                                     attach_labels=attach_labels,
                                                     z_table=z_table,
                                                    max_episode_steps=n_max_steps,
                                                    valid_action_masks=valid_action_masks,
                                                    embed_hydrogens=EMBED_HYDROGENS,
                                                    pocket_feature_type=pocket_feature_type,)
                                  for _ in range(batch_size)]

memory = {}
best_scores = {i: score for i, score in enumerate(initial_scores)}

batch_env = BatchEnv(envs,
                     memory,
                     best_scores,
                     pocket_feature_type=pocket_feature_type,
                     scoring_function=SCORING_FUNCTION,
                     )

agent = Agent(protected_fragments=protected_fragments,
              atomic_num_table=z_table,
              features_dim=batch_env.pocket_feature_dim,
              pocket_feature_type=pocket_feature_type,
              )
state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_graph_04_07_2024_14_36_06_step_240000_ent_1.0.pt')
agent.load_state_dict(state_dict)
agent = agent.to(device)

gen_mols_dir = '/home/bb596/hdd/ymir/generated_mols_random/'
if not os.path.exists(gen_mols_dir):
    os.mkdir(gen_mols_dir)
    
for complx_i, complx in enumerate(tqdm(complexes)):
    ligand_filename = selected_ligand_filenames[complx_i]
    target_dirname, real_ligand_filename = ligand_filename.split('/')
    
    current_complexes = [complx for _ in range(batch_size)]
    current_generation_paths = [[] for _ in range(batch_size)]
    real_data_idxs = []
                
    generated_ligands = []
    
    for _ in range(n_gen_molecules // batch_size):
        seed_idxs = [seed_i
                     for seed_i, c_i in enumerate(seed2complex) 
                     if c_i == complx_i]
        seed_idxs = np.random.choice(seed_idxs, size=batch_size, replace=True)
        current_seeds = [seeds[seed_i] for seed_i in seed_idxs]
        current_initial_scores = [initial_scores[seed_idx] for seed_idx in seed_idxs]
        current_native_scores = [native_scores[seed_idx] for seed_idx in seed_idxs]
    
        with torch.no_grad():
            next_info = batch_env.reset(current_complexes,
                                    current_seeds,
                                    current_initial_scores,
                                    current_native_scores,
                                    seed_idxs,
                                    real_data_idxs,
                                    current_generation_paths)
            n_envs = batch_size
            next_terminated = [False] * n_envs
            
            step_i = 0
            while step_i < n_max_steps and not all(next_terminated):
            
                current_terminated = next_terminated
                current_masks = batch_env.get_valid_action_mask()
                
                current_frag_actions = []
                for mask in current_masks:
                    possible_frag_idxs = mask.nonzero().squeeze()
                    current_frag_actions.append(np.random.choice(possible_frag_idxs))
                
                t = batch_env.step(frag_actions=current_frag_actions)
                
                step_rewards, next_terminated, next_truncated = t
                
                step_i += 1
            
        generated_ligands.extend([env.seed.mol for env in batch_env.envs])
        
    # Save generated ligands
    save_dir = os.path.join(gen_mols_dir, target_dirname)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, f'generated_{real_ligand_filename}.sdf')
    with Chem.SDWriter(save_path) as writer:
        for mol in generated_ligands:
            writer.write(mol)
    
    


# test_crossdocked = CrossDocked(subset='test')
# ligand_filenames = test_crossdocked.get_ligand_filenames()
# for ligand_filename in tqdm(ligand_filenames):
#     target_dirname, real_ligand_filename = ligand_filename.split('/')
#     native_ligand = test_crossdocked.get_native_ligand(ligand_filename)
#     # native_ligand = Chem.AddHs(native_ligand, addCoords=True)
            
#     original_structure_path = test_crossdocked.get_original_structure_path(ligand_filename)
#     pdb_id = real_ligand_filename[:4]
#     complx = Complex(ligand=native_ligand,
#                      protein_path=original_structure_path)
#     fragments, frags_mol_atom_mapping = get_fragments_from_mol(native_ligand)
    
#     if len(fragments) > 1:
    
#         has_7 = [7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()] for fragment in fragments]
        
#         if not any(has_7):
    
#             initial_scores = []
#             smina_cli = SminaCLI()
#             ligands = [fragment.mol for fragment in fragments]
#             initial_scores = smina_cli.get(receptor_paths=[complx.vina_protein.pdbqt_filepath for _ in ligands],
#                                 ligands=ligands)
            
#             smina_cli = SminaCLI()
#             scores = smina_cli.get(receptor_paths=[complx.vina_protein.pdbqt_filepath],
#                                 ligands=[complx.ligand])
#             native_score = scores[0]
            
#             generated_ligands = []
#             for i in range(n_gen_molecules // batch_size):
#                 # batch_idxs = range(i * batch_size, (i + 1) * batch_size)
#                 seed_choice = np.random.choice(len(fragments), size=batch_size, replace=True)
#                 current_seeds = [fragments[seed_idx] for seed_idx in seed_choice]
#                 current_complexes = [complx for _ in range(batch_size)]
#                 current_native_scores = [native_score for _ in range(batch_size)]
#                 current_initial_scores = [initial_scores[seed_idx] for seed_idx in seed_choice]
#                 current_generation_paths = [[] for _ in range(batch_size)]
#                 real_data_idxs = []
                
#                 with torch.no_grad():
#                     next_info = batch_env.reset(current_complexes,
#                                             current_seeds,
#                                             current_initial_scores,
#                                             current_native_scores,
#                                             seed_choice,
#                                             real_data_idxs,
#                                             current_generation_paths)
#                     n_envs = batch_size
#                     next_terminated = [False] * n_envs
                    
#                     step_i = 0
#                     while step_i < n_max_steps and not all(next_terminated):
                    
#                         current_terminated = next_terminated
#                         current_obs = batch_env.get_obs()
                        
#                         current_masks = batch_env.get_valid_action_mask()
                        
#                         batch = Batch.from_data_list(current_obs)
#                         batch = batch.to(device)
#                         features = agent.extract_features(batch)
                        
#                         # if current_masks.size()[0] != features.size()[0] :
#                         #     import pdb;pdb.set_trace()
                        
#                         current_masks = current_masks.to(device)
#                         if pocket_feature_type == 'soap':
#                             b = None
#                         else:
#                             b = batch.batch
#                             if features.shape[0] != b.shape[0]:
#                                 import pdb;pdb.set_trace()
#                         current_policy: CategoricalMasked = agent.get_policy(features,
#                                                                                 batch=b,
#                                                                                 masks=current_masks)
#                         current_action: Action = agent.get_action(current_policy)
#                         current_frag_actions = current_action.frag_i.cpu()  
                        
#                         t = batch_env.step(frag_actions=current_frag_actions)
                        
#                         step_rewards, next_terminated, next_truncated = t
                        
#                         step_i += 1
                    
#                 generated_ligands.extend([env.seed.mol for env in batch_env.envs])
                
#             # Save generated ligands
#             with Chem.SDWriter(f'/home/bb596/hdd/ymir/generated_cross_docked_late/{real_ligand_filename}_generated.sdf') as writer:
#                 for mol in generated_ligands:
#                     writer.write(mol)
            
    # import pdb;pdb.set_trace()