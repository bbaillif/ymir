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
                         SCORING_FUNCTION)
from ymir.utils.fragment import get_fragments_from_mol
from ymir.utils.train import get_paths, ppo_episode, reinforce_episode
from ymir.data.structure import GlideProtein, Complex
from ymir.metrics.activity.glide_score import GlideScore
from ymir.metrics.activity.smina_cli import SminaCLI


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
n_envs = 140
batch_size = n_envs
n_max_steps = 5 # number of maximum fragment growing
# lr = 5e-4
# lr = 1e-4
lr = 5e-6
gamma = 0.95 # discount factor for rewards
device = torch.device('cuda')
ent_coef = 2.0
# ent_coef = 0.2
# ent_patience = 10
ent_patience = 100
ent_coef_step = ent_coef / 20
gae_lambda = 0.95
clip_coef = 0.5
n_epochs = 5
vf_coef = 0.5
max_grad_value = 0.5

rl_algorithm = 'reinforce'
assert rl_algorithm in ['ppo', 'reinforce']

use_entropy_loss = True
pocket_feature_type = 'soap'
assert pocket_feature_type in ['graph', 'soap']

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_v2_ppo_{timestamp}"

writer = SummaryWriter(f"logs_reinforce_v2/{experiment_name}")

fragment_library = FragmentLibrary(removeHs=False)

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
    
with Chem.SDWriter('protected_fragments.sdf') as sdwriter:
    for f in protected_fragments:
        sdwriter.write(f.mol)
    
torsion_angles_deg = TORSION_ANGLES_DEG
n_torsions = len(torsion_angles_deg)
rotated_fragments = get_rotated_fragments(protected_fragments, torsion_angles_deg)

if len(set(synthon_smiles)) != len(synthon_smiles):
    import pdb;pdb.set_trace()

pocket_radius = POCKET_RADIUS

lp_pdbbind = pd.read_csv('LP_PDBBind.csv')
lp_pdbbind = lp_pdbbind.rename({'Unnamed: 0' : 'PDB_ID'}, axis=1)

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)

glide_not_working = ['1px4', '2hjb', '3c2r', '3pcg', '3pce', '3pcn', '4bps']

lp_pdbbind_subsets = {pdb_id: subset 
                      for pdb_id, subset in zip(lp_pdbbind['PDB_ID'], lp_pdbbind['new_split'])}

complexes: list[Complex] = []
seeds: list[Fragment] = []
generation_sequences = []
z = list(zip(protein_paths, ligands))
# z = z[:500]
for protein_path, ligand in tqdm(z, total=len(z)):
    fragments, frags_mol_atom_mapping = get_fragments_from_mol(ligand)
    
    if len(fragments) > 1:
        
        has_7 = [7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()] for fragment in fragments]
        
        if not any(has_7):
        
            fragment_in_actions = []
            for fragment in fragments:
                frag_copy = Fragment.from_fragment(fragment)
                
                # up_smiles = Chem.MolToSmiles(frag_copy.mol)
                
                s_smiles_list = []
                authorized = []
                for attach_point, label in frag_copy.get_attach_points().items():
                    p_frag = Fragment.from_fragment(fragment)
                    p_frag.unprotect()
                    atom_ids_to_keep = [atom_id for atom_id in range(p_frag.mol.GetNumAtoms())]
                    atom_ids_to_keep.remove(attach_point)
                    p_frag.protect(atom_ids_to_keep=atom_ids_to_keep)
                    s_smiles = Chem.MolToSmiles(p_frag.mol)
                    in_dataset = False
                    if s_smiles in synthon_smiles:
                        smiles_idx = synthon_smiles.index(s_smiles)
                        if label in attach_labels[smiles_idx]:
                            in_dataset = True
                    authorized.append(in_dataset)
                
                fragment_in_actions.append(all(authorized))
            
            # take ligands with fragments only in the protected fragments
            if all(fragment_in_actions):
                current_seeds, current_paths, current_complexes  = get_paths(ligand, 
                                                                            fragments, 
                                                                            protein_path, 
                                                                            lp_pdbbind_subsets, 
                                                                            glide_not_working,
                                                                            frags_mol_atom_mapping,
                                                                            synthon_smiles,
                                                                            attach_labels)
                seeds.extend(current_seeds)
                complexes.extend(current_complexes)
                generation_sequences.extend(current_paths)
            # elif sum(fragment_in_actions) + 1 == len(fragment_in_actions):
            #     import pdb;pdb.set_trace()
                
n_complexes = len(complexes)

initial_scores_path = f'/home/bb596/hdd/ymir/initial_scores_{n_complexes}cplx_{n_fragments}frags_{n_torsions}rots_{SCORING_FUNCTION}_min.pkl'
native_scores_path = f'/home/bb596/hdd/ymir/native_scores_{n_complexes}cplx_{n_fragments}frags_{n_torsions}rots_{SCORING_FUNCTION}_min.pkl'
if (not os.path.exists(initial_scores_path)) or (not os.path.exists(native_scores_path)):

    initial_scores = []
    native_scores = []
    # Non optimal
    for seed, complx in tqdm(zip(seeds, complexes), total=n_complexes):
        smina_cli = SminaCLI()
        seed_copy = Fragment.from_fragment(seed)
        seed_copy.protect()
        mols = [seed_copy.mol]
        scores = smina_cli.get(receptor_paths=[complx.vina_protein.pdbqt_filepath],
                            ligands=mols)
        assert len(scores) == 1
        initial_scores.append(scores[0])
        
        smina_cli = SminaCLI(score_only=False)
        scores = smina_cli.get(receptor_paths=[complx.vina_protein.pdbqt_filepath],
                            ligands=[complx.ligand])
        assert len(scores) == 1
        native_scores.append(scores[0])
        
    with open(initial_scores_path, 'wb') as f:
        pickle.dump(initial_scores, f)
        
    with open(native_scores_path, 'wb') as f:
        pickle.dump(native_scores, f)
    
else:
    with open(initial_scores_path, 'rb') as f:
        initial_scores = pickle.load(f)
        
    with open(native_scores_path, 'rb') as f:
        native_scores = pickle.load(f)


assert len(initial_scores) == n_complexes
assert len(native_scores) == n_complexes

# remove complexes with positive initial Glide scores
original_valid_idxs = [i for i, score in enumerate(initial_scores) if score < 0]
valid_idxs = original_valid_idxs[:560]

seeds = [seeds[i] for i in valid_idxs]
complexes = [complexes[i] for i in valid_idxs]
generation_sequences = [generation_sequences[i] for i in valid_idxs]
initial_scores = [initial_scores[i] for i in valid_idxs]
native_scores = [native_scores[i] for i in valid_idxs]

valid_action_masks = get_masks(attach_labels)

n_complexes = len(complexes)
logging.info(f'Training on {n_complexes} complexes')

logging.info(f'There are {len(rotated_fragments)} fragments with {len(rotated_fragments[0])} torsions each')

envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(rotated_fragments=rotated_fragments,
                                                     attach_labels=attach_labels,
                                                     z_table=z_table,
                                                    max_episode_steps=n_max_steps,
                                                    valid_action_masks=valid_action_masks,
                                                    embed_hydrogens=EMBED_HYDROGENS,
                                                    pocket_feature_type=pocket_feature_type,)
                                  for _ in range(n_envs)]

n_torsions = len(TORSION_ANGLES_DEG)
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
                     pocket_feature_type=pocket_feature_type,
                     scoring_function=SCORING_FUNCTION,
                     )

agent = Agent(protected_fragments=protected_fragments,
              atomic_num_table=z_table,
              features_dim=batch_env.pocket_feature_dim,
              pocket_feature_type=pocket_feature_type,
              )
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_v1_11_04_2024_20_12_31_6000.pt')
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
        
        # seed_i = episode_i % n_complexes
        # seed_idxs = [seed_i] * n_envs
        seed_idxs = [idx % n_complexes for idx in range(episode_i * n_envs, (episode_i + 1) * n_envs)]
            
        if (0 in seed_idxs) and n_real_data > 0:
            n_real_data -= 1
            
        if rl_algorithm == 'ppo':
            ppo_episode(agent,
                    episode_i,
                    seed_idxs, 
                    seeds,
                    complexes,
                    initial_scores,
                    generation_sequences,
                    batch_env,
                    n_max_steps,
                    n_real_data,
                    device,
                    gamma,
                    gae_lambda,
                    n_epochs,
                    writer,
                    optimizer,
                    max_grad_value,
                    vf_coef,
                    ent_coef,
                    batch_size,
                    clip_coef,
                    use_entropy_loss)
            
        else:
            scores = reinforce_episode(agent,
                            episode_i,
                            seed_idxs, 
                            seeds,
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
            
        current_scores.extend(scores)
        assert len(current_scores) <= len(complexes)
        if len(current_scores) == len(complexes):
            mean_score = np.mean(current_scores)
        
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                logging.info(f'New best mean score: {best_mean_score}, after {iteration_since_best} steps')
                iteration_since_best = 0
            else:
                iteration_since_best += 1
                if iteration_since_best > ent_patience:
                    logging.info(f'Decreasing entropy coefficient from {ent_coef} to {max(0.00, ent_coef - ent_coef_step)} after {iteration_since_best} steps without improvement')
                    ent_coef = max(0.00, ent_coef - ent_coef_step)
                    iteration_since_best = 0
            
            current_scores = []
        
        # Save memory every memory_save_step
        # memory_size = len(memory)
        # current_memory_i = memory_size // memory_save_step
        # if (current_memory_i > memory_i):
        #     with open(memory_path, 'wb') as f:
        #         pickle.dump(memory, f)
        #     memory_i = current_memory_i
        # logging.info(f'Memory size: {memory_size}')
        
        if ((episode_i + 1) % 5000 == 0):
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