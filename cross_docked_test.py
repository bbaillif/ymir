import numpy as np
import torch
import os
import pickle
import time

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
                         CROSSDOCKED_POCKET10_PATH,
                         POCKET_FEATURE_TYPE)

n_gen_molecules = 100
batch_size = 25
n_max_steps = 10
pocket_feature_type = POCKET_FEATURE_TYPE
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

n_fragments = 200
# Remove fragment having at least one heavy atom not in list
restricted_fragments = fragment_library.get_restricted_fragments(z_list, 
                                                                max_attach=5, 
                                                                max_torsions=5,
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

smina_dir = '/home/bb596/ymir/smina_test'
if not os.path.exists(smina_dir):
    os.mkdir(smina_dir)

smina_ligands_dir = '/home/bb596/ymir/smina_test/ligands'
if not os.path.exists(smina_ligands_dir):
    os.mkdir(smina_ligands_dir)

smina_output_dir = '/home/bb596/ymir/smina_test/poses'
if not os.path.exists(smina_output_dir):
    os.mkdir(smina_output_dir)

batch_env = BatchEnv(envs,
                     memory,
                     best_scores,
                     smina_ligands_dir=smina_ligands_dir,
                     smina_output_directory=smina_output_dir,
                     pocket_feature_type=pocket_feature_type,
                     scoring_function=SCORING_FUNCTION,
                     )

agent = Agent(protected_fragments=protected_fragments,
              atomic_num_table=z_table,
              features_dim=batch_env.pocket_feature_dim,
              pocket_feature_type=pocket_feature_type,
              )
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_graph_17_07_2024_23_39_59_step_10000_ent_0.1.pt') # single
state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_graph_19_07_2024_22_20_26_step_5000_ent_0.1.pt') # multi
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_graph_23_07_2024_20_17_26_step_4000_ent_0.1.pt') # le
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_graph_25_07_2024_18_27_05_step_5000_ent_0.1.pt') # complexV0.1
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_graph_12_08_2024_15_26_30_step_5000_ent_0.1.pt')

agent.load_state_dict(state_dict)
agent = agent.to(device)

# model_prefix = 'ymir_vrds'
model_suffix = 'ymir_vr'

# gen_mols_dir = '/home/bb596/hdd/ymir/generated_mols_cd_ppo_multi/'
# gen_mols_dir = '/home/bb596/hdd/ymir/generated_mols_cd_ppo_complex/'
gen_mols_dir = f'/home/bb596/hdd/ymir/generated_mols_{model_suffix}/'
if not os.path.exists(gen_mols_dir):
    os.mkdir(gen_mols_dir)
    
gen_times = []
    
for complx_i, complx in enumerate(tqdm(complexes)):
    ligand_filename = selected_ligand_filenames[complx_i]
    target_dirname, real_ligand_filename = ligand_filename.split('/')
    
    current_complexes = [complx for _ in range(batch_size)]
    current_generation_paths = [[] for _ in range(batch_size)]
    real_data_idxs = []
                
    generated_ligands = []
    
    start_time = time.time()
    
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
                current_obs = batch_env.get_obs()
                
                current_masks = batch_env.get_valid_action_mask()
                
                batch = Batch.from_data_list(current_obs)
                batch = batch.to(device)
                features = agent.extract_features(batch)
                
                # if current_masks.size()[0] != features.size()[0] :
                #     import pdb;pdb.set_trace()
                
                current_masks = current_masks.to(device)
                if pocket_feature_type == 'soap':
                    b = None
                else:
                    b = batch.batch
                    if features.shape[0] != b.shape[0]:
                        import pdb;pdb.set_trace()
                current_policy: CategoricalMasked = agent.get_policy(features,
                                                                        batch=b,
                                                                        masks=current_masks)
                current_action: Action = agent.get_action(current_policy)
                current_frag_actions = current_action.frag_i.cpu()  
                
                t = batch_env.step(frag_actions=current_frag_actions)
                
                step_rewards, next_terminated, next_truncated = t
                
                step_i += 1
            
        generated_ligands.extend([env.seed.mol for env in batch_env.envs])
        
    gen_times.append(time.time() - start_time)
        
    # Save generated ligands
    save_dir = os.path.join(gen_mols_dir, target_dirname)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, f'generated_{real_ligand_filename}.sdf')
    with Chem.SDWriter(save_path) as writer:
        for mol in generated_ligands:
            writer.write(mol)
    
print(f'Average generation time: {np.mean(gen_times)}')
with open(f'/home/bb596/hdd/ymir/gen_times_{model_suffix}.txt', 'w') as f:
    for gen_time in gen_times:
        f.write(f'{gen_time}\n')