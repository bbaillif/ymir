import time
import torch
import random
import numpy as np
import pandas as pd
import logging
import pickle
import os
# import wandb

from tqdm import tqdm
from rdkit import Chem
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from ymir.fragment_library import FragmentLibrary
from ymir.utils.fragment import (center_fragments, 
                                 get_rotated_fragments,
                                 get_masks)

from ymir.atomic_num_table import AtomicNumberTable

from ymir.env import (FragmentBuilderEnv, 
                            BatchEnv)
from ymir.policy import Agent
from ymir.data import Fragment
from ymir.params import (EMBED_HYDROGENS, 
                         SEED,
                         POCKET_RADIUS,
                         TORSION_ANGLES_DEG,
                         SCORING_FUNCTION,
                         CROSSDOCKED_POCKET10_PATH,
                         SMINA_LIGANDS_DIRECTORY,
                         SMINA_OUTPUT_DIRECTORY)
from ymir.utils.fragment import get_fragments_from_mol
from ymir.utils.train import get_paths, ppo_episode, reinforce_episode
from ymir.data.structure import GlideProtein, Complex
from ymir.metrics.activity.glide_score import GlideScore
from ymir.metrics.activity.smina_cli import SminaCLI
from ymir.data.cross_docked import CrossDocked
from multiprocessing import Pool


logging.basicConfig(filename='train.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode="w")

seed = SEED
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# 1 episode = grow fragments + update NN
n_episodes = 1_000_000
# n_envs = 256 # we will have protein envs in parallel
n_envs = 64
batch_size = n_envs
n_max_steps = 10 # number of maximum fragment growing
lr = 1e-5
# lr = 1e-5
gamma = 0.95 # discount factor for rewards
device = torch.device('cuda')
ent_coef = 0.1
# ent_coef = 0.2
# ent_patience = 10
ent_patience = 5
ent_decrease = False
ent_coef_step = ent_coef / 20
gae_lambda = 0.95
clip_coef = 0.5
n_epochs = 5
vf_coef = 0.5
max_grad_value = 0.5

use_entropy_loss = True

rl_algorithm = 'ppo'
assert rl_algorithm in ['ppo', 'reinforce']

pocket_feature_type = 'graph'
assert pocket_feature_type in ['graph', 'soap']

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_{pocket_feature_type}_{timestamp}"

writer = SummaryWriter(f"logs_final/{experiment_name}")

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

z_list = [0, 6, 7, 8, 16, 17]
if EMBED_HYDROGENS:
    z_list.append(1)
z_table = AtomicNumberTable(zs=z_list)

# Remove ligands having at least one heavy atom not in list
# And add hydrogens
ligands = fragment_library.get_restricted_ligands(z_list)
if SCORING_FUNCTION == 'glide':
    ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]

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
    
with Chem.SDWriter('protected_fragments.sdf') as sdwriter:
    for f in protected_fragments:
        sdwriter.write(f.mol)
    
torsion_angles_deg = TORSION_ANGLES_DEG
n_torsions = len(torsion_angles_deg)
rotated_fragments = get_rotated_fragments(protected_fragments, torsion_angles_deg)

if len(set(synthon_smiles)) != len(synthon_smiles):
    import pdb;pdb.set_trace()

pocket_radius = POCKET_RADIUS

cd_split = cross_docked_train.get_split()

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

complexes: list[Complex] = []
seeds: list[Fragment] = []
seed2complex = []
generation_sequences = []
protein_paths = []
considered_ligands = []
pocket_paths = []
for duo, ligand in tqdm(zip(cd_split, cd_ligands), total=len(cd_split)):
    if all([atom.GetAtomicNum() in z_list for atom in ligand.GetAtoms()]):
        pocket_path, ligand_subpath = duo
        protein_paths.append(cross_docked_train.get_original_structure_path(ligand_filename=ligand_subpath))
        considered_ligands.append(ligand)
        pocket_paths.append(pocket_path)
        
training_data_path = '/home/bb596/hdd/ymir/training_data.pkl'
# if not os.path.exists(training_data_path):
        
    # with Pool(20, maxtasksperchild=1) as p:
    #     results = p.map(get_seeds_complx, 
    #                     [(protein_path, ligand) for protein_path, ligand in zip(protein_paths, considered_ligands)])
    #     for result in tqdm(results, total=len(protein_paths)):
    #         seeds.extend(result[0])
    #         seed2complex.extend([len(complexes) - 1] * len(result[0]))
    #         generation_sequences.extend([None] * len(result[0]))
    #         complexes.append(result[1])

        
for protein_path, pocket_path, ligand in tqdm(zip(protein_paths, pocket_paths, considered_ligands), 
                                                total=len(protein_paths)):
    current_seeds, complx, pocket = get_seeds_complx((protein_path, pocket_path, ligand))
    if len(current_seeds) > 0:
        if any([seed.mol.GetNumAtoms() > 6 for seed in current_seeds]):
            complexes.append(complx)
        for current_seed in current_seeds:
            if current_seed.mol.GetNumAtoms() > 6:
                seeds.append(current_seed)
                seed2complex.append(len(complexes) - 1)
                generation_sequences.append(None)
        # complexes.append(complx)
        # seeds.extend(current_seeds)
        # seed2complex.extend([len(complexes) - 1 for _ in current_seeds])
        # generation_sequences.extend([None for _ in current_seeds])
           
    # training_data = {'seeds': seeds,
    #                     'seed2complex': seed2complex,
    #                     'generation_sequences': generation_sequences,
    #                     'complexes': complexes}
            
    # with open(training_data_path, 'wb') as f:
    #     pickle.dump(training_data, f)
         
# else:
#     with open(training_data_path, 'rb') as f:
#         training_data = pickle.load(f)
#     seeds = training_data['seeds']
#     seed2complex = training_data['seed2complex']
#     generation_sequences = training_data['generation_sequences']
#     complexes = training_data['complexes']
    
        # protein_path = cross_docked_train.get_original_structure_path(ligand_filename=ligand_subpath)
        # fragments, frags_mol_atom_mapping = get_fragments_from_mol(ligand)
        
        # if len(fragments) > 1:
            
        #     if not all([7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()] for fragment in fragments]):
            
        #         try:
        #             complx = Complex(ligand, protein_path)
        #             # if all([atom.GetAtomicNum() in z_list for atom in complx.pocket.mol.GetAtoms()]):
        #             pf = complx.vina_protein.pdbqt_filepath
        #             pocket = complx.pocket
        #             complexes.append(complx)
            
        #             for fragment in fragments:
        #                 if not any([7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()]]):
        #                     current_seed = Fragment.from_fragment(fragment)
        #                     seeds.append(current_seed)
        #                     seed2complex.append(len(complexes) - 1)
        #                     generation_sequences.append(None)
        #             # else:
        #             #     import pdb;pdb.set_trace()
        #             #     print(f'No pocket for {ligand_subpath}')
        #         except:
        #             print(f'No pdbqt file for {ligand_subpath}')
                
n_complexes = len(complexes)

# for complx in complexes:
# def get_vina_protein(i: int,
#                      complx: Complex):
#     try:
#         pf = complx.vina_protein.pdbqt_filepath
#     except Exception as e:
#         print(type(e))
#         print(str(e))
# with Pool(12) as p:
#     results = p.starmap_async(get_vina_protein, [(i, complx) for i, complx in enumerate(complexes)])
#     for result in tqdm(results.get(), total=len(complexes)):
#         pass

initial_scores_path = f'/home/bb596/hdd/ymir/initial_scores_{n_complexes}cplx_200frags_3rots_{SCORING_FUNCTION}_min_cd.pkl'
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

native_scores_path = f'/home/bb596/hdd/ymir/native_scores_{n_complexes}cplx_200frags_3rots_{SCORING_FUNCTION}_min_cd.pkl'
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

# remove complexes with positive initial Glide scores
original_valid_idxs = [i for i, score in enumerate(initial_scores) if score < 0]
n_valid = len(original_valid_idxs)
valid_idxs = original_valid_idxs[:n_valid // n_envs * n_envs]

seeds = [seeds[i] for i in valid_idxs]
seed2complex = [seed2complex[i] for i in valid_idxs]
generation_sequences = [generation_sequences[i] for i in valid_idxs]
initial_scores = [initial_scores[i] for i in valid_idxs]
native_scores = [native_scores[i] for i in valid_idxs]

valid_action_masks = get_masks(attach_labels)

n_complexes = len(set(seed2complex))
n_seeds = len(seeds)
logging.info(f'Training on {n_complexes} complexes for {n_seeds} seeds')

logging.info(f'There are {len(rotated_fragments)} fragments with {len(rotated_fragments[0])} torsions each')



envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(rotated_fragments=rotated_fragments,
                                                     attach_labels=attach_labels,
                                                     z_table=z_table,
                                                    max_episode_steps=n_max_steps,
                                                    valid_action_masks=valid_action_masks,
                                                    embed_hydrogens=EMBED_HYDROGENS,
                                                    pocket_feature_type=pocket_feature_type,)
                                  for _ in range(n_envs)]

# memory_path = f'/home/bb596/hdd/ymir/memory_{len(original_valid_idxs)}cplx_{n_fragments}frags_{n_torsions}rots_{SCORING_FUNCTION}_min.pkl'
# if os.path.exists(memory_path):
#     with open(memory_path, 'rb') as f:
#         memory = pickle.load(f)
# else:
#     memory = {}
memory = {}
    
memory_size = len(memory)

memory_save_step = 500
memory_i = memory_size // memory_save_step

best_scores = {i: score for i, score in enumerate(initial_scores)}

batch_env = BatchEnv(envs,
                     memory,
                     best_scores,
                     smina_ligands_dir=SMINA_LIGANDS_DIRECTORY,
                     smina_output_directory=SMINA_OUTPUT_DIRECTORY,
                     pocket_feature_type=pocket_feature_type,
                     scoring_function=SCORING_FUNCTION,
                     )

agent = Agent(protected_fragments=protected_fragments,
              atomic_num_table=z_table,
              features_dim=batch_env.pocket_feature_dim,
              pocket_feature_type=pocket_feature_type,
              )
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_graph_02_07_2024_16_24_49_step_40000_ent_1.0.pt')
# agent.load_state_dict(state_dict)

agent = agent.to(device)

optimizer = Adam(agent.parameters(), lr=lr)

# n_real_data = n_envs // 2 + 1
# n_real_data = 8
n_real_data = 0

best_mean_score = 0
iteration_since_best = 0
current_scores = []

try:

    for episode_i in tqdm(range(n_episodes)):
        
        logging.debug(f'Episode i: {episode_i}')
        
        # seed_i = (episode_i + 0) % n_seeds # + 40000 from the save
        # seed_idxs = [seed_i] * n_envs
        seed_idxs = [idx % len(seeds) 
                     for idx in range(episode_i * n_envs, (episode_i + 1) * n_envs)]
            
        if (0 in seed_idxs) and n_real_data > 0:
            n_real_data -= 1
            
        if rl_algorithm == 'ppo':
            ppo_episode(agent,
                        episode_i,
                        seed_idxs, 
                        seeds,
                        seed2complex,
                        complexes,
                        initial_scores,
                        native_scores,
                        generation_sequences,
                        batch_env,
                        n_max_steps,
                        n_real_data,
                        device,
                        gamma,
                        writer,
                        optimizer,
                        ent_coef,
                        use_entropy_loss,
                        pocket_feature_type,
                        gae_lambda,
                        n_epochs,
                        max_grad_value,
                        vf_coef,
                        batch_size,
                        clip_coef)
            
        else:
            scores = reinforce_episode(agent,
                            episode_i,
                            seed_idxs, 
                            seeds,
                            seed2complex,
                            complexes,
                            initial_scores,
                            native_scores,
                            generation_sequences,
                            batch_env,
                            n_max_steps,
                            n_real_data,
                            device,
                            gamma,
                            writer,
                            optimizer,
                            ent_coef,
                            use_entropy_loss,
                            pocket_feature_type)
            
        # current_scores.extend(scores)
        # assert len(current_scores) <= len(seeds)
        # if len(current_scores) == len(seeds):
        #     mean_score = np.mean(current_scores)
        
        #     if mean_score < best_mean_score:
        #         best_mean_score = mean_score
        #         logging.info(f'New best mean score: {best_mean_score}, after {iteration_since_best} steps')
        #         iteration_since_best = 0
        #     else:
        #         iteration_since_best += 1
        #         if ent_decrease and iteration_since_best > ent_patience:
        #             logging.info(f'Decreasing entropy coefficient from {ent_coef} to {max(0.00, ent_coef - ent_coef_step)} after {iteration_since_best} steps without improvement')
        #             ent_coef = max(0.00, ent_coef - ent_coef_step)
        #             iteration_since_best = 0
            
        #     current_scores = []
        
        # Save memory every memory_save_step
        # memory_size = len(memory)
        # current_memory_i = memory_size // memory_save_step
        # if (current_memory_i > memory_i):
        #     with open(memory_path, 'wb') as f:
        #         pickle.dump(memory, f)
        #     memory_i = current_memory_i
        # logging.info(f'Memory size: {memory_size}')
        
        if ((episode_i + 1) % 1000 == 0):
            if use_entropy_loss:
                save_path = f'/home/bb596/hdd/ymir/models/{experiment_name}_step_{(episode_i + 1)}_ent_{ent_coef}.pt'
                # ent_coef = max(0.00, ent_coef - ent_coef_step)
            else:
                save_path = f'/home/bb596/hdd/ymir/models/{experiment_name}_step_{(episode_i + 1)}.pt'
            torch.save(agent.state_dict(), save_path)
            # with open(memory_path, 'wb') as f:
            #     pickle.dump(memory, f)
            
            
        # import pdb;pdb.set_trace()
            
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
    
# except Exception as exc:
#     print(type(exc))
#     print(exc)
#     import pdb;pdb.set_trace()
#     print(exc)
        
# with open(memory_path, 'wb') as f:
#     pickle.dump(memory, f)
# memory_i = current_memory_i
        
# except Exception as e:
#     print(e)
#     import pdb;pdb.set_trace()
#     print(e)
        
# agent = Agent(*args, **kwargs)
# agent.load_state_dict(torch.load(PATH))
# agent.eval()