import time
import torch
import random
import numpy as np
import pandas as pd
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
from ymir.data.structure import Complex
from ymir.utils.fragment import (get_seeds, 
                                 center_fragments, 
                                 get_masks,
                                 select_mol_with_symbols,
                                 ConstructionSeed)

from ymir.atomic_num_table import AtomicNumberTable

from ymir.env_reinforce import (FragmentBuilderEnv, 
                            BatchEnv)
from ymir.policy import Agent, Action
from ymir.data import Fragment
from ymir.params import (EMBED_HYDROGENS, 
                         HIDDEN_IRREPS,
                         SEED,
                         VINA_DATASET_PATH)
from ymir.metrics.activity import VinaScore, VinaScorer
from ymir.metrics.activity.vina_cli import VinaCLI
from regressor import SOAPFeaturizer
from sklearn.feature_selection import VarianceThreshold

logging.basicConfig(filename='train_reinforce.log', 
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
n_episodes = 100_000
n_envs = 256 # we will have protein envs in parallel
batch_size = min(n_envs, 256) # NN batch, input is Data, output are actions + predicted reward
n_steps = 10 # number of maximum fragment growing
n_epochs = 5 # number of times we update the network per episode
lr = 1e-5
gamma = 0.95 # discount factor for rewards
gae_lambda = 0.95 # lambda factor for GAE
device = torch.device('cuda')
clip_coef = 0.5
ent_coef = 0.1
# ent_coef = 0.01
vf_coef = 0.5
max_grad_value = 0.5

n_complexes = 200
use_entropy_loss = True

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_v1_{timestamp}"

writer = SummaryWriter(f"logs_reinforce/{experiment_name}")

removeHs = not EMBED_HYDROGENS
fragment_library = FragmentLibrary(removeHs=removeHs)
# ligands = fragment_library.ligands
# protected_fragments = fragment_library.protected_fragments

# ligands = [ligand 
#            for ligand in ligands 
#            if ligand.GetNumHeavyAtoms() < 50]

# random.shuffle(ligands)

z_list = [0, 6, 7, 8, 16, 17]
if EMBED_HYDROGENS:
    z_list.append(1)
z_table = AtomicNumberTable(zs=z_list)

# Remove ligands having at least one heavy atom not in list
ligands = fragment_library.get_restricted_ligands(z_list)

# Remove fragment having at least one heavy atom not in list
protected_fragments = fragment_library.get_restricted_fragments(z_list)
           
protected_fragments_smiles = [Chem.MolToSmiles(frag) for frag in protected_fragments]

n_samples = 2500
input_soaps_path = f'/home/bb596/hdd/ymir/input_soaps_{n_samples}_reinforce.p'

pre_computed_soap = os.path.exists(input_soaps_path)
# pre_computed_soap = False

if pre_computed_soap:
    with open(input_soaps_path, 'rb') as f:
        input_soaps = pickle.load(f)
else:
    input_soaps = []
    soap_featurizer = SOAPFeaturizer()

dataset_path = '/home/bb596/hdd/ymir/dataset/'
data_filenames = sorted(os.listdir(dataset_path))


lp_pdbbind = pd.read_csv('LP_PDBBind.csv')
lp_pdbbind = lp_pdbbind.rename({'Unnamed: 0' : 'PDB_ID'}, axis=1)

all_seeds = []
all_constructs = []
complexes = []
all_scores = []
native_scores = []
subsets = []
for data_filename in tqdm(data_filenames[:n_samples]):
    
    data_filepath = os.path.join(dataset_path, data_filename)
    with open(data_filepath, 'rb') as f:
        data = pickle.load(f)
    
    protein_path = data['protein_path']
    protein_path = protein_path.replace('.pdbqt', '.pdb')
    ligand = data['ligand']
    ligand_name = ligand.GetProp('_Name')
    pdb_id = ligand_name.split('_')[0]
    if pdb_id in lp_pdbbind['PDB_ID'].values:
        subset = lp_pdbbind[lp_pdbbind['PDB_ID'] == pdb_id]['new_split'].values[0]
        if subset in ['train', 'val', 'test']:
            removed_fragment_atom_idxs = data['removed_fragment_atom_idxs']
            absolute_scores = data['absolute_scores']
            native_score = data['native_score']
            
            seed = ConstructionSeed(ligand, removed_fragment_atom_idxs)
            complx = Complex(ligand, protein_path)

            construct, removed_fragment, bond = seed.decompose()
            attach_points = construct.get_attach_points()
            attach_point = list(attach_points.keys())[0]
            attach_label = list(attach_points.values())[0]
            
            center_pos = construct.GetConformer().GetPositions()[attach_point]

            if not pre_computed_soap:
                soap = soap_featurizer.featurize_complex(construct, complx.pocket.mol, center_pos)
                input_soaps.append(soap)

            all_seeds.append(seed)
            all_constructs.append(construct)
            complexes.append(complx)
            all_scores.append(absolute_scores)
            native_scores.append(native_score)
            subsets.append(subset)
    
if not pre_computed_soap:
    with open(input_soaps_path, 'wb') as f:
        pickle.dump(input_soaps, f)
    
input_soaps = np.array(input_soaps)
vt = VarianceThreshold()
selected_soaps = vt.fit_transform(input_soaps)

input_soaps = torch.tensor(selected_soaps, dtype=torch.float)
    
train_seed_idxs = [i for i, subset in enumerate(subsets) if subset == 'train']
val_seed_idxs = [i for i, subset in enumerate(subsets) if subset == 'val']
train_size = len(train_seed_idxs)
val_size = len(val_seed_idxs)
print(f'Train size: {train_size}')
print(f'Val size: {val_size}')

center_fragments(protected_fragments)
final_fragments = protected_fragments

valid_action_masks = get_masks(final_fragments)

logging.info(f'There are {len(final_fragments)} fragments')

envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(protected_fragments=final_fragments,
                                                     z_table=z_table,
                                                    max_episode_steps=n_steps,
                                                    valid_action_masks=valid_action_masks,
                                                    embed_hydrogens=EMBED_HYDROGENS)
                                  for _ in range(n_envs)]
state_action = tuple[int, int] # (seed_i, action_i)
memory: dict[state_action, float] = {}
batch_env = BatchEnv(envs,
                     memory)

n_val_envs = len(val_seed_idxs)
val_envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(protected_fragments=final_fragments,
                                                     z_table=z_table,
                                                    max_episode_steps=n_steps,
                                                    valid_action_masks=valid_action_masks,
                                                    embed_hydrogens=EMBED_HYDROGENS)
                                        for _ in range(n_val_envs)]
val_batch_env = BatchEnv(val_envs,
                        memory)

agent = Agent(protected_fragments=final_fragments,
              atomic_num_table=z_table,
              features_dim=input_soaps.shape[-1])
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_v1_28_02_2024_18_20_19_8000.pt')
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_v1_11_04_2024_20_12_31_6000.pt')

# agent.load_state_dict(state_dict)

agent = agent.to(device)

optimizer = Adam(agent.parameters(), lr=lr)

n_complexes = len(complexes)
logging.info(f'Training on {n_complexes} complexes')

logging.info('Start RL')


def episode(seed_idxs, 
            batch_env: BatchEnv,
            train: bool = True):
    current_constructs = [all_constructs[seed_i] for seed_i in seed_idxs]
    current_complexes = [complexes[seed_i] for seed_i in seed_idxs]
    current_absolute_scores = [all_scores[seed_i] for seed_i in seed_idxs]
    current_native_scores = [native_scores[seed_i] for seed_i in seed_idxs]
    current_soaps = input_soaps[seed_idxs]
    
    next_info = batch_env.reset(current_complexes,
                                current_constructs,
                                initial_scores=current_native_scores,
                                absolute_scores=current_absolute_scores,
                                )
    next_terminated = [False] * n_envs
    
    step_i = 0
    while step_i < n_steps and not all(next_terminated):
        
        current_obs = current_soaps.to(device)
        
        current_masks = batch_env.get_valid_action_mask()
        
        features = current_obs
        current_masks = current_masks.to(device)
        current_action: Action = agent.get_action(features,
                                                    masks=current_masks)
        current_frag_actions = current_action.frag_i.cpu()
        current_frag_logprobs = current_action.frag_logprob.cpu()
        
        t = batch_env.step(frag_actions=current_frag_actions)
        
        logging.info(current_frag_actions)
        logging.info(current_frag_logprobs.exp())
        
        reward, next_terminated, next_truncated, next_info = t
        
        step_i += 1
        
    logging.info(np.around(reward, 2))  
    reward = torch.tensor(reward, device=device, dtype=torch.float)
    
    policy_loss = -current_action.frag_logprob * reward
    loss = policy_loss
    if use_entropy_loss:
        entropy = current_action.frag_entropy
        entropy_loss = - entropy * ent_coef # we want to keep entropy high = minimize negative entropy
        loss = loss + entropy_loss

    if train:
        subset = 'train'
    else:
        subset = 'val'
    writer.add_scalar(f"{subset}/policy_loss", policy_loss.mean().item(), episode_i)
    writer.add_scalar(f"{subset}/fragment_entropy", entropy.mean().item(), episode_i)
    writer.add_scalar(f"{subset}/entropy_loss", entropy_loss.mean().item(), episode_i)
    writer.add_scalar(f"{subset}/loss", loss.mean().item(), episode_i)
    writer.add_scalar(f"{subset}/mean_reward", reward.mean().item(), episode_i)

    return loss

# try:

for episode_i in tqdm(range(n_episodes)):
    
    logging.debug(f'Episode i: {episode_i}')
    
    start_idx = episode_i * n_envs % train_size
    end_idx = (episode_i + 1) * n_envs % train_size
    if start_idx > end_idx:
        seed_is = list(range(0, end_idx)) + list(range(start_idx, train_size))
    else:
        seed_is = list(range(start_idx, end_idx))
    train_i = [idx for i, idx in enumerate(train_seed_idxs) if i in seed_is]
    logging.info(train_i)
    
    loss = episode(train_i, batch_env)
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    
    with torch.no_grad():
        episode(val_seed_idxs, val_batch_env, train=False)
    
    # if ((episode_i + 1) % 500 == 0):
    #     import pdb;pdb.set_trace()
    
    if ((episode_i + 1) % 2000 == 0):
        save_path = f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt'
        torch.save(agent.state_dict(), f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt')
            
# except KeyboardInterrupt:
#     import pdb;pdb.set_trace()
        
# except Exception as e:
#     print(e)
#     import pdb;pdb.set_trace()
#     print(e)
        
# agent = Agent(*args, **kwargs)
# agent.load_state_dict(torch.load(PATH))
# agent.eval()