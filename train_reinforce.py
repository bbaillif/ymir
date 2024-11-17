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
                                 get_rotated_fragments,
                                 get_masks,
                                 select_mol_with_symbols,
                                 ConstructionSeed)

from ymir.atomic_num_table import AtomicNumberTable

from ymir.env import (FragmentBuilderEnv, 
                            BatchEnv)
from ymir.policy import Agent, Action
from ymir.data import Fragment
from ymir.params import (EMBED_HYDROGENS, 
                         HIDDEN_IRREPS,
                         SEED,
                         VINA_DATASET_PATH,
                         POCKET_RADIUS,
                         NEIGHBOR_RADIUS,
                         TORSION_ANGLES_DEG)
from ymir.metrics.activity import VinaScore, VinaScorer
from ymir.metrics.activity.vina_cli import VinaCLI
from regressor import SOAPFeaturizer, GraphFeaturizer
from sklearn.feature_selection import VarianceThreshold
from ymir.utils.fragment import get_fragments_from_mol
from collections import deque
from torch_cluster import radius_graph
from ymir.save_state import StateSave, Memory

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
n_envs = 8 # we will have protein envs in parallel
batch_size = min(n_envs, 16) # NN batch, input is Data, output are actions + predicted reward
n_steps = 10 # number of maximum fragment growing
# lr = 5e-4
lr = 2e-4
gamma = 0.95 # discount factor for rewards
device = torch.device('cuda')
# ent_coef = 0.25
# ent_coef = 0.1
ent_coef = 0.10

use_entropy_loss = False

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_v1_{timestamp}"

writer = SummaryWriter(f"logs_reinforce/{experiment_name}")

# removeHs = not EMBED_HYDROGENS
fragment_library = FragmentLibrary(removeHs=False)
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
ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]
# print([ligand.GetProp('_Name') for ligand in ligands[:50]])
# ligands = ligands[30:]
ligands = random.sample(ligands, 10)

# Remove fragment having at least one heavy atom not in list
protected_fragments = fragment_library.get_restricted_fragments(z_list, max_attach=2, n_fragments=100)
# protected_fragments = protected_fragments[:100]

protected_fragments_smiles = [Chem.MolToSmiles(frag.to_mol()) for frag in protected_fragments]

pocket_radius = POCKET_RADIUS
# input_data_list_path = f'/home/bb596/hdd/ymir/data/input_data_{pocket_radius}.p'

# pre_computed_data = os.path.exists(input_data_list_path)
# # pre_computed_soap = False

# if pre_computed_data:
#     with open(input_data_list_path, 'rb') as f:
#         input_data_list = pickle.load(f)
# else:
#     input_data_list = []
#     graph_featurizer = GraphFeaturizer(z_table=z_table)

# dataset_path = '/home/bb596/hdd/ymir/dataset/'
# data_filenames = sorted(os.listdir(dataset_path))

lp_pdbbind = pd.read_csv('LP_PDBBind.csv')
lp_pdbbind = lp_pdbbind.rename({'Unnamed: 0' : 'PDB_ID'}, axis=1)

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)

complexes = []
seeds = []
for protein_path, ligand in tqdm(zip(protein_paths, ligands), total=len(protein_paths)):
    fragments, frags_mol_atom_mapping = get_fragments_from_mol(ligand)
    if len(fragments) > 1:
        ligand_name = ligand.GetProp('_Name')
        pdb_id = ligand_name.split('_')[0]
        if pdb_id in lp_pdbbind['PDB_ID'].values:
            subset = lp_pdbbind[lp_pdbbind['PDB_ID'] == pdb_id]['new_split'].values[0]
            if subset == 'train':
                complx = Complex(ligand, protein_path)
                # break
                complexes.extend([complx for _ in fragments])
                seeds.extend(fragments)

# complexes = [complx for _ in fragments]
# seeds = fragments

receptor_paths = [complx.vina_protein.pdbqt_filepath 
                    for complx in complexes]
native_ligands = [complx.ligand 
                    for complx in complexes]
protected_seeds = [Fragment.from_fragment(seed) for seed in seeds]
# pseeds_h = []
mols = []
for pseed in protected_seeds:
    pseed.protect()
    mols.append(pseed.to_mol())
    # pseed = Chem.RemoveHs(pseed)
    # pseed_h = Chem.AddHs(pseed)
    # pseeds_h.append(pseed_h)
    
vina_cli = VinaCLI()
initial_scores = vina_cli.get(receptor_paths=receptor_paths,
                            native_ligands=native_ligands,
                            ligands=mols)
        
# train_seed_idxs = [i for i, subset in enumerate(subsets) if subset == 'train']
# val_seed_idxs = [i for i, subset in enumerate(subsets) if subset == 'val']

# train_seed_idxs = train_seed_idxs[:100]
# val_seed_idxs = val_seed_idxs[:100]

# train_size = len(train_seed_idxs)
# val_size = len(val_seed_idxs)
# print(f'Train size: {train_size}')
# print(f'Val size: {val_size}')

center_fragments(protected_fragments)

torsion_angles_deg = TORSION_ANGLES_DEG
final_fragments = get_rotated_fragments(protected_fragments, torsion_angles_deg)

valid_action_masks = get_masks(final_fragments)

logging.info(f'There are {len(final_fragments)} fragments')

envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(protected_fragments=final_fragments,
                                                     z_table=z_table,
                                                    max_episode_steps=n_steps,
                                                    valid_action_masks=valid_action_masks,
                                                    embed_hydrogens=EMBED_HYDROGENS)
                                  for _ in range(n_envs)]

memory_path = '/home/bb596/hdd/ymir/memory_100cplx_100frags_36rots.pkl'
if os.path.exists(memory_path):
    with open(memory_path, 'rb') as f:
        memory = pickle.load(f)
else:
    memory: Memory = {}
    
memory_size = len(memory)

memory_save_step = 500
memory_i = memory_size // memory_save_step

batch_env = BatchEnv(envs,
                     memory)


# val_seed_idxs = val_seed_idxs[:100]
# n_val_envs = len(val_seed_idxs)
# val_envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(protected_fragments=final_fragments,
#                                                      z_table=z_table,
#                                                     max_episode_steps=n_steps,
#                                                     valid_action_masks=valid_action_masks,
#                                                     embed_hydrogens=EMBED_HYDROGENS)
#                                         for _ in range(n_val_envs)]
# val_batch_env = BatchEnv(val_envs,
#                         memory)

agent = Agent(protected_fragments=final_fragments,
              atomic_num_table=z_table,
            #   features_dim=len(all_scores[0])
              )
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_v1_28_02_2024_18_20_19_8000.pt')
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_v1_11_04_2024_20_12_31_6000.pt')

# agent.load_state_dict(state_dict)

agent = agent.to(device)

optimizer = Adam(agent.parameters(), lr=lr)

n_complexes = len(complexes)
logging.info(f'Training on {n_complexes} complexes')

def episode(seed_idxs, 
            batch_env: BatchEnv,
            train: bool = True):
    
    # try:
    fragment_features = agent.extract_fragment_features()
    
    current_seeds = [seeds[seed_i] for seed_i in seed_idxs]
    current_complexes = [complexes[seed_i] for seed_i in seed_idxs]
    current_initial_scores = [initial_scores[seed_i] for seed_i in seed_idxs]
    
    next_info = batch_env.reset(current_complexes,
                                current_seeds,
                                current_initial_scores,
                                seed_idxs)
    next_terminated = [False] * n_envs
    
    step_i = 0
    ep_logprobs = []
    ep_rewards = []
    ep_entropies = []
    ep_terminateds: list[list[bool]] = [] # (n_steps, n_envs)
    while step_i < n_steps and not all(next_terminated):
        
        current_terminated = next_terminated
        current_obs = batch_env.get_obs()
        
        current_masks = batch_env.get_valid_action_mask()
        
        batch = Batch.from_data_list(current_obs)
        batch = batch.to(device)
        
        radius = NEIGHBOR_RADIUS
        edge_src, edge_dst = radius_graph(batch.pos, 
                                          radius,
                                          batch=batch.batch)
        if edge_src.unique().size()[0] != (batch.x.size()[0]):
            import pdb;pdb.set_trace()
        if edge_dst.unique().size()[0] != (batch.x.size()[0]):
            import pdb;pdb.set_trace()
        
        features = agent.extract_features(batch)
        
        if current_masks.size()[0] != features.size()[0] :
            import pdb;pdb.set_trace()
        
        current_masks = current_masks.to(device)
        current_action: Action = agent.get_action(features,
                                                  fragment_features,
                                                    masks=current_masks)
        current_frag_actions = current_action.frag_i.cpu()
        current_frag_logprobs = current_action.frag_logprob.cpu()
        
        t = batch_env.step(frag_actions=current_frag_actions)
        
        logging.info(current_frag_actions)
        logging.info(current_frag_logprobs.exp())
        
        step_rewards, next_terminated, next_truncated, next_info = t
        
        ep_logprobs.append(current_action.frag_logprob)
        ep_rewards.append(step_rewards)
        ep_entropies.append(current_action.frag_entropy)
        ep_terminateds.append(current_terminated)
        
        step_i += 1
        
    batch_env.save_state()
        
    reversed_rewards = reversed(ep_rewards)
    reversed_terminateds = reversed(ep_terminateds)
    reversed_returns = [] # (n_non_term_envs, 1)
    z = zip(reversed_terminateds, reversed_rewards)
    for step_terminated, step_rewards in z:
        step_non_terminated = [not terminated for terminated in step_terminated]
        non_terminated_idxs = np.where(step_non_terminated)[0]
        
        try:
            assert len(non_terminated_idxs) == len(step_rewards)
        except:
            import pdb;pdb.set_trace()
        
        current_return = []
        last_returns = {env_i: 0 for env_i in range(n_envs)}
        z = zip(non_terminated_idxs, step_rewards)
        for env_i, step_reward in z:
            last_return = last_returns[env_i]
            retrn = step_reward + last_return * gamma
            current_return.append(retrn)
            last_returns[env_i] = retrn

        current_return = torch.tensor(current_return, device=device)
        reversed_returns.append(current_return)
        
    returns = list(reversed(reversed_returns))
    
    all_returns = torch.cat(returns)
    all_logprobs = torch.cat(ep_logprobs)
    all_entropies = torch.cat(ep_entropies)
    
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-5)
    
    logging.info(all_returns)
    
    policy_loss = -all_logprobs * all_returns
    if use_entropy_loss:
        entropy_loss = - all_entropies * ent_coef
        loss = policy_loss + entropy_loss
    else:
        loss = policy_loss
        
    loss = loss.mean()

    # n_obs = all_returns.size()[0]
    # inds = np.arange(n_obs)
    # np.random.shuffle(inds)
    # all_losses = []
    # for start in range(0, n_obs, batch_size):
    #     end = start + batch_size
    #     minibatch_inds = inds[start:end]
        
    #     n_sample = len(minibatch_inds)
    #     if n_sample > 2: # we only want to update if we have more than 1 sample in the batch
    
    #         mb_logprobs = all_logprobs[minibatch_inds]
    #         mb_returns = all_returns[minibatch_inds]
    #         mb_entropies = all_entropies[minibatch_inds]
    
    #         policy_loss = -mb_logprobs * mb_returns
    #         if use_entropy_loss:
    #             entropy_loss = - mb_entropies * ent_coef
    #             loss = policy_loss + entropy_loss
    #         else:
    #             loss = policy_loss
            
    #         loss = loss.mean()
    
            # # losses = []
            # policy_losses = []
            # if use_entropy_loss:
            #     entropy_losses = []
            # for logprob, retrn, entropy in zip(all_logprobs, all_returns, all_entropies):
            #     policy_loss = -logprob * retrn
            #     policy_losses.append(policy_loss)
            #     # loss = policy_loss
            #     if use_entropy_loss:
            #         entropy_loss = - entropy * ent_coef # we want to keep entropy high = minimize negative entropy
            #         entropy_losses.append(entropy_loss)
            #         # loss = loss + entropy_loss
            #     # losses.append(loss)
                
            # loss = torch.cat(policy_losses).sum()
            # if use_entropy_loss:
            #     loss = loss + torch.cat(entropy_losses).sum() * ent_coef
        
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # # logging.info(np.around(reward, 2))  
    # R = 0
    # returns = deque()
    # for r in ep_rewards[::-1]:
    #     R = r + gamma * R
    #     returns.appendleft(R)
    # returns = torch.tensor(returns, device=device, dtype=torch.float)
    # # reward = torch.tensor(reward, device=device, dtype=torch.float)

    if train:
        subset = 'train'
    else:
        subset = 'val'
        
    writer.add_scalar(f"{subset}/policy_loss", policy_loss.mean().item(), episode_i)
    # if use_entropy_loss:
    # writer.add_scalar(f"{subset}/mean_entropy", all_entropies.mean().item(), episode_i)
    writer.add_scalar(f"{subset}/loss", loss.item(), episode_i)
    writer.add_scalar(f"{subset}/fragment_entropy", all_entropies.mean().item(), episode_i)
    writer.add_scalar(f"{subset}/mean_reward", returns[0].mean().item(), episode_i)

    # return all_losses
        
    # except Exception as e:
    #     print(type(e))
    #     print(e)
    #     import pdb;pdb.set_trace()
        

# try:

for episode_i in tqdm(range(n_episodes)):
    
    logging.debug(f'Episode i: {episode_i}')
    
    # start_idx = episode_i * n_envs % train_size
    # end_idx = (episode_i + 1) * n_envs % train_size
    # if start_idx > end_idx:
    #     seed_is = list(range(0, end_idx)) + list(range(start_idx, train_size))
    # else:
    #     seed_is = list(range(start_idx, end_idx))
    # train_i = [idx for i, idx in enumerate(train_seed_idxs) if i in seed_is]
    # logging.info(train_i)
    
    train_i = []
    current_i = 0
    while len(train_i) < n_envs:
        train_i.append(current_i)
        current_i = (current_i + 1) % n_complexes
        
    episode(train_i, batch_env)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    
    # Save memory every memory_save_step
    memory_size = len(memory)
    current_memory_i = memory_size // memory_save_step
    if (current_memory_i > memory_i):
        with open(memory_path, 'wb') as f:
            pickle.dump(memory, f)
        memory_i = current_memory_i
    
    # with torch.no_grad():
    #     episode(val_seed_idxs, val_batch_env, train=False)
    
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