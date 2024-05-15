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
                                 ConstructionSeed,
                                 get_neighbor_id_for_atom_id)

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
                         TORSION_ANGLES_DEG,
                         SCORING_FUNCTION)
from ymir.metrics.activity import VinaScore, VinaScorer
from ymir.metrics.activity.vina_cli import VinaCLI
from regressor import SOAPFeaturizer, GraphFeaturizer
from sklearn.feature_selection import VarianceThreshold
from ymir.utils.fragment import get_fragments_from_mol
from collections import deque
from torch_cluster import radius_graph
from ymir.save_state import StateSave, Memory
from ymir.data.structure import GlideProtein
from ymir.distribution import CategoricalMasked
from ymir.molecule_builder import potential_reactions

logging.basicConfig(filename='train_ppo.log', 
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
n_envs = 1 # we will have protein envs in parallel
batch_size = min(n_envs, 16) # NN batch, input is Data, output are actions + predicted reward
n_max_steps = 5 # number of maximum fragment growing
# lr = 5e-4
lr = 2e-4
gamma = 0.95 # discount factor for rewards
device = torch.device('cuda')
# ent_coef = 0.25
# ent_coef = 0.1
ent_coef = 0.10
gae_lambda = 0.95
clip_coef = 0.5
n_epochs = 5
vf_coef = 0.5
max_grad_value = 0.5

use_entropy_loss = True

# n_ligands = 200
n_fragments = 100

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_v2_ppo_{timestamp}"

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
# ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]
# print([ligand.GetProp('_Name') for ligand in ligands[:50]])
# ligands = ligands[30:]
# ligands = random.sample(ligands, n_ligands)

# Remove fragment having at least one heavy atom not in list
protected_fragments = fragment_library.get_restricted_fragments(z_list, 
                                                                max_attach=3, 
                                                                max_torsions=0,
                                                                n_fragments=n_fragments)
# for fragment in protected_fragments:
#     Chem.SanitizeMol(fragment.to_mol())
    
with Chem.SDWriter('protected_fragments.sdf') as sdwriter:
    for f in protected_fragments:
        sdwriter.write(f.to_mol())
# import pdb;pdb.set_trace()
    
protected_fragments_smiles = [Chem.MolToSmiles(frag.to_mol()) for frag in protected_fragments]

unprotected_fragments = []
for fragment in protected_fragments:
    frag_copy = Fragment.from_fragment(fragment)
    frag_copy.unprotect()
    unprotected_fragments.append(frag_copy)

unprotected_fragments_smiles = [Chem.MolToSmiles(frag.to_mol()) for frag in unprotected_fragments]

smiles_combos = [(p_smiles, up_smiles) 
                 for p_smiles, up_smiles in zip(protected_fragments_smiles, unprotected_fragments_smiles)]

pocket_radius = POCKET_RADIUS

lp_pdbbind = pd.read_csv('LP_PDBBind.csv')
lp_pdbbind = lp_pdbbind.rename({'Unnamed: 0' : 'PDB_ID'}, axis=1)

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)

glide_not_working = ['1px4', '2hjb', '3c2r']

complexes = []
seeds = []
generation_sequences = []
for protein_path, ligand in tqdm(zip(protein_paths, ligands), total=len(protein_paths)):
    fragments, frags_mol_atom_mapping = get_fragments_from_mol(ligand)
    
    if len(fragments) > 1:
        
        fragment_in_actions = []
        for fragment in fragments:
            frag_copy = Fragment.from_fragment(fragment)
            up_smiles = Chem.MolToSmiles(frag_copy.to_mol())
            
            p_smiles_list = []
            for attach_point, label in frag_copy.get_attach_points().items():
                p_frag = Fragment.from_fragment(fragment)
                p_frag.protect(atom_ids_to_keep=[attach_point])
                p_smiles = Chem.MolToSmiles(p_frag.to_mol())
                p_smiles_list.append(p_smiles)
            
            up_smiles_ok = up_smiles in unprotected_fragments_smiles
            p_smiles_ok = all([p_smiles in protected_fragments_smiles for p_smiles in p_smiles_list])
            
            # if up_smiles_ok and not p_smiles_ok:
            #     import pdb;pdb.set_trace()
            # combo = (p_smiles, up_smiles)
            fragment_in_actions.append(up_smiles_ok and p_smiles_ok)
        
        # take ligands with fragments only in the protected fragments
        if sum(fragment_in_actions) == len(fragment_in_actions):
        
            ligand_name = ligand.GetProp('_Name')
            pdb_id = ligand_name.split('_')[0]
            if (pdb_id in lp_pdbbind['PDB_ID'].values) and (not pdb_id in glide_not_working):
                subset = lp_pdbbind[lp_pdbbind['PDB_ID'] == pdb_id]['new_split'].values[0]
                if subset == 'train':
                    complx = Complex(ligand, protein_path)
                    try:
                        glide_protein = GlideProtein(pdb_filepath=complx.vina_protein.protein_clean_filepath,
                                                        native_ligand=complx.ligand)
                    except KeyboardInterrupt:
                        import pdb;pdb.set_trace()
                    except:
                        logging.info(f'No Glide structure for {pdb_id}')
                    else:
                        # complexes.extend([complx for _ in fragments])
                        
                        def get_master_frag_i(fragment: Fragment, 
                                              atom_id_to_keep: int):
                            frag_copy = Fragment.from_fragment(fragment)
                            up_smiles = Chem.MolToSmiles(frag_copy.to_mol())
                            # if up_smiles == '[1*]C([6*])=O':
                            #     import pdb;pdb.set_trace()
                            frag_copy.protect([atom_id_to_keep])
                            p_smiles = Chem.MolToSmiles(frag_copy.to_mol())
                            combo = (p_smiles, up_smiles)
                            idxs = [i for i, smiles_combo in enumerate(smiles_combos) if smiles_combo == combo]
                            assert len(idxs) == 1
                            try:
                                master_frag_i = smiles_combos.index(combo)
                            except Exception as e:
                                print(str(e))
                                import pdb;pdb.set_trace()
                            
                            return master_frag_i
                        
                        def find_broken_bond(ligand, seed, seed_i, fragment, fragment_i, frags_mol_atom_mapping):
                            try:
                                for attach_point, label in seed.get_attach_points().items():
                                    neigh_id1 = get_neighbor_id_for_atom_id(seed.to_mol(), attach_point)
                                    native_neigh_id1 = frags_mol_atom_mapping[seed_i][neigh_id1]
                                    for attach_point2, label2 in fragment.get_attach_points().items():
                                        neigh_id2 = get_neighbor_id_for_atom_id(fragment.to_mol(), attach_point2)
                                        native_neigh_id2 = frags_mol_atom_mapping[fragment_i][neigh_id2]
        
                                        potential_reactions_seed = potential_reactions[label]
                                        if label2 in potential_reactions_seed:
                                            bond = ligand.GetBondBetweenAtoms(native_neigh_id1, native_neigh_id2)
                                            if bond is not None:
                                                ap1_position = seed.to_mol().GetConformer().GetPositions()[attach_point]
                                                neigh2_position = fragment.to_mol().GetConformer().GetPositions()[neigh_id2]
                                                if ap1_position.tolist() == neigh2_position.tolist():
                                                    return (neigh_id1, attach_point, neigh_id2, attach_point2)
                            except Exception as e:
                                print(str(e))
                                import pdb;pdb.set_trace()
                            return None
                        
                        rg = list(range(len(fragments)))
                        for initial_frag_i in rg:
                            placed_fragments_i = [initial_frag_i]
                            # master_frag_i = get_master_frag_i(current_fragment)
                            action_sequence = []
                            
                            while len(action_sequence) != (len(fragments) - 1):
                                current_frags_i = list(placed_fragments_i)
                                for frag_i in current_frags_i:
                                    current_fragment = fragments[frag_i]
                                    for other_frag_i in rg:
                                        other_fragment = fragments[other_frag_i]
                                        if other_frag_i not in placed_fragments_i:
                                            broken_bond = find_broken_bond(ligand, current_fragment, frag_i, 
                                                                           other_fragment, other_frag_i, frags_mol_atom_mapping)
                                            if broken_bond:
                                                neigh_id1, attach_point, neigh_id2, attach_point2 = broken_bond
                                                attach_position = current_fragment.to_mol().GetConformer().GetPositions()[attach_point]
                                                master_frag_i = get_master_frag_i(other_fragment, atom_id_to_keep=attach_point2)
                                                # if master_frag_i == 4 and ligand_name == '6t1j_ligand':
                                                #     import pdb;pdb.set_trace()
                                                action = (attach_position, master_frag_i)
                                                action_sequence.append(action)
                                                placed_fragments_i.append(other_frag_i)
                            
                            if ligand_name == '1ajp_ligand':
                                import pdb;pdb.set_trace()
                            
                            seeds.append(fragments[initial_frag_i])
                            generation_sequences.append(action_sequence)
                            complexes.append(complx)

n_complexes = len(complexes)

receptor_paths = [complx.vina_protein.pdbqt_filepath 
                    for complx in complexes]

initial_scores = [0 for _ in receptor_paths]

center_fragments(protected_fragments)

torsion_angles_deg = TORSION_ANGLES_DEG
n_torsions = len(torsion_angles_deg)
final_fragments = get_rotated_fragments(protected_fragments, torsion_angles_deg)

valid_action_masks = get_masks(final_fragments)

logging.info(f'There are {len(final_fragments)} fragments')

envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(protected_fragments=final_fragments,
                                                     z_table=z_table,
                                                    max_episode_steps=n_max_steps,
                                                    valid_action_masks=valid_action_masks,
                                                    embed_hydrogens=EMBED_HYDROGENS)
                                  for _ in range(n_envs)]

memory_path = f'/home/bb596/hdd/ymir/memory_{n_complexes}cplx_{n_fragments}frags_36rots_glide_min.pkl'
if os.path.exists(memory_path):
    with open(memory_path, 'rb') as f:
        memory = pickle.load(f)
else:
    memory = {}

# memory = {}
    
memory_size = len(memory)

memory_save_step = 500
memory_i = memory_size // memory_save_step

# memory = {}

batch_env = BatchEnv(envs,
                     memory
                     )


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

agent = Agent(protected_fragments=protected_fragments,
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

def training_loop(agent: Agent,
                  b_action_obs: list[Data],
                  b_pocket_obs: list[Data],
                    b_frag_actions,
                    b_frag_logprobs,
                    b_advantages,
                    b_returns,
                    b_values,
                    b_masks,
                    n_epochs=5,
                    ):
    n_obs = len(b_action_obs)
    logging.info(f'{n_obs} obs for epochs')
    
    inds = np.arange(n_obs)
    for epoch_i in range(n_epochs):
        logging.debug(f'Epoch i: {epoch_i}')
        np.random.shuffle(inds)
        for start in range(0, n_obs, batch_size):
            end = start + batch_size
            minibatch_inds = inds[start:end]
            
            n_sample = len(minibatch_inds)
            if n_sample > 2: # we only want to update if we have more than 1 sample in the batch
            
                mb_advantages = b_advantages[minibatch_inds]
                mb_advantages = mb_advantages.to(device)
                
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                mb_action_obs = [b_action_obs[i] for i in minibatch_inds]
                mb_pocket_obs = [b_pocket_obs[i] for i in minibatch_inds]
                mb_frag_actions = b_frag_actions[minibatch_inds]
                mb_masks = b_masks[minibatch_inds]
                
                # batch = Batch.from_data_list(mb_action_obs)
                batch = mb_action_obs[0]
                batch = batch.to(device)
                
                pocket_batch = mb_pocket_obs[0]
                pocket_batch = pocket_batch.to(device)
                
                features = agent.extract_features(batch)
                pocket_features = agent.extract_features(pocket_batch)
                
                current_masks = mb_masks.to(device)
                current_frag_actions = mb_frag_actions.to(device)
                current_policy = agent.get_policy(features=features,
                                                    masks=current_masks,)
                current_action = agent.get_action(current_policy,
                                                    frag_actions=current_frag_actions)
                
                current_frag_logprobs = current_action.frag_logprob
                current_frag_entropy = current_action.frag_entropy
                
                mb_frag_logprobs = b_frag_logprobs[minibatch_inds]
                mb_frag_logprobs = mb_frag_logprobs.to(device)
                frag_ratio = (current_frag_logprobs - mb_frag_logprobs).exp()
                
                frag_approx_kl = (mb_frag_logprobs - current_frag_logprobs).mean()
                
                frag_pg_loss1 = -mb_advantages * frag_ratio
                frag_pg_loss2 = -mb_advantages * torch.clamp(frag_ratio, 
                                                        min=1 - clip_coef, 
                                                        max=1 + clip_coef)
                frag_pg_loss = torch.max(frag_pg_loss1, frag_pg_loss2).mean()
                frag_entropy_loss = current_frag_entropy.mean()
                
                mb_returns = b_returns[minibatch_inds]
                mb_values = b_values[minibatch_inds]
                mb_returns = mb_returns.to(device)
                mb_values = mb_values.to(device)
                current_values = agent.get_value(features=pocket_features)
                v_loss_unclipped = ((current_values - mb_returns) ** 2)
                v_clipped = mb_values + torch.clamp(current_values - mb_values, 
                                                    -clip_coef, 
                                                    clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = v_loss_max.mean()
                
                loss = frag_pg_loss + (v_loss * vf_coef) 
                if use_entropy_loss:
                    loss = loss - (ent_coef * frag_entropy_loss)
                
                optimizer.zero_grad()
                loss.backward()
                
                writer.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in agent.actor_tp.parameters()]), 
                             global_step=episode_i)
                # import pdb;pdb.set_trace()
                
                nn.utils.clip_grad_value_(parameters=agent.parameters(), 
                                        clip_value=max_grad_value)
                optimizer.step()
                
    writer.add_scalar("train/value_loss", v_loss.item(), episode_i)
    writer.add_scalar("train/policy_loss", frag_pg_loss.item(), episode_i)
    writer.add_scalar("train/entropy", frag_entropy_loss.mean().item(), episode_i)
    writer.add_scalar("train/approx_kl", frag_approx_kl.item(), episode_i)
    writer.add_scalar("train/loss", loss.item(), episode_i)
    # writer.add_scalar("train/mean_return", b_returns.mean(), episode_i)


def episode(seed_idxs, 
            batch_env: BatchEnv,
            train: bool = True,
            n_real_data: int = 0):
    
    # try:
    
    current_seeds = [seeds[seed_i] for seed_i in seed_idxs]
    current_complexes = [complexes[seed_i] for seed_i in seed_idxs]
    current_initial_scores = [initial_scores[seed_i] for seed_i in seed_idxs]
    
    if n_real_data > 0:
        real_data_idxs = np.random.choice(range(len(seed_idxs)), n_real_data, replace=False)
    else:
        real_data_idxs = []
        
    current_generation_paths = [generation_sequences[seed_i] for seed_i in seed_idxs]
    
    next_info = batch_env.reset(current_complexes,
                                current_seeds,
                                current_initial_scores,
                                seed_idxs,
                                real_data_idxs,
                                current_generation_paths)
    next_terminated = [False] * n_envs
    
    with torch.no_grad():
        ep_logprobs = []
        # ep_rewards = []
        # ep_entropies = []
        ep_action_obs = []
        ep_pocket_obs = []
        ep_actions = []
        ep_masks = []
        ep_values = []
        ep_terminateds: list[list[bool]] = [] # (n_steps, n_envs)
        termination_step = [n_max_steps] * n_envs
        
        step_i = 0
        while step_i < n_max_steps and not all(next_terminated):
            
            current_terminated = next_terminated
            current_obs = batch_env.get_action_obs()
            ep_action_obs.append(current_obs)
            
            current_pocket = batch_env.get_pocket_obs()
            ep_pocket_obs.append(current_pocket)
            
            current_masks = batch_env.get_valid_action_mask()
            ep_masks.append(current_masks)
            
            action_batch = current_obs[0]
            action_batch = action_batch.to(device)
            
            pocket_batch = current_pocket[0]
            pocket_batch = pocket_batch.to(device)
            
            action_features = agent.extract_features(action_batch)
            pocket_features = agent.extract_features(pocket_batch)
            
            if current_masks.size()[0] != action_features.size()[0] :
                import pdb;pdb.set_trace()
            
            current_masks = current_masks.to(device)
            current_policy: CategoricalMasked = agent.get_policy(action_features,
                                                                    masks=current_masks)
            current_action: Action = agent.get_action(current_policy)
            current_frag_actions = current_action.frag_i.cpu()
            current_frag_logprobs = current_action.frag_logprob.cpu()
            
            ep_actions.append(current_frag_actions)
            
            t = batch_env.step(frag_actions=current_frag_actions,
                               real_data_idxs=real_data_idxs)
            
            logging.info(current_frag_actions)
            logging.info(current_frag_logprobs.exp())
            
            next_terminated, next_truncated = t
            
            for env_i, (prev_term, new_term) in enumerate(zip(current_terminated, next_terminated)):
                if prev_term == False and new_term == True:
                    termination_step[env_i] = step_i
            
            ep_logprobs.append(current_frag_logprobs)
            # ep_rewards.append(step_rewards)
            # ep_entropies.append(current_action.frag_entropy)
            ep_terminateds.append(current_terminated)
            
            current_values = agent.get_value(pocket_features)
            ep_values.append(current_values.cpu())
            
            step_i += 1
            
            import pdb;pdb.set_trace()
            
        assert all(next_terminated)
            
        if SCORING_FUNCTION == 'glide':
            final_rewards = batch_env.get_rewards_glide(next_truncated, 
                                                        mininplace=True
                                                        )
        else:
            final_rewards = batch_env.get_rewards(next_truncated)
        batch_env.save_state(reverse_transformations=True)
        
        ep_rewards = []
        n_steps_in_episode = step_i
        for step_i in range(n_steps_in_episode):
            step_rewards = []
            for env_i in range(n_envs):
                if step_i < termination_step[env_i]:
                    step_rewards.append(0)
                elif step_i == termination_step[env_i]:
                    step_rewards.append(final_rewards[env_i])
            ep_rewards.append(step_rewards)
            
        reversed_values = reversed(ep_values)
        reversed_rewards = reversed(ep_rewards)
        reversed_terminateds = reversed(ep_terminateds)
        reversed_returns = [] # (n_non_term_envs, 1)
        reversed_advantages = [] # (n_non_term_envs, 1)
        last_values = {env_i: 0 for env_i in range(n_envs)}
        lastgaelams = {env_i: 0 for env_i in range(n_envs)}
        z = zip(reversed_values, reversed_terminateds, reversed_rewards)
        for step_values, step_terminated, step_rewards in z:
            step_non_terminated = [not terminated for terminated in step_terminated]
            non_terminated_idxs = np.where(step_non_terminated)[0]
            
            try:
                assert len(non_terminated_idxs) == len(step_rewards)
            except:
                import pdb;pdb.set_trace()
            
            current_retrn = []
            current_advantage = []
            z = zip(non_terminated_idxs, step_values, step_rewards)
            for env_i, step_value, step_reward in z:
                last_value = last_values[env_i]
                delta = step_reward + last_value * gamma - step_value
                
                lastgaelam = lastgaelams[env_i]
                advantage = delta + gamma * gae_lambda * lastgaelam
                
                last_values[env_i] = step_value
                lastgaelams[env_i] = advantage
                
                retrn = advantage + step_value
                current_retrn.append(retrn)
                # advantage = retrn - step_value
                current_advantage.append(advantage)

            current_retrn = torch.stack(current_retrn)
            current_advantage = torch.stack(current_advantage)
            reversed_returns.append(current_retrn)
            reversed_advantages.append(current_advantage)
            
        returns = list(reversed(reversed_returns))
        advantages = list(reversed(reversed_advantages))
        
        b_action_obs: list[Data] = []
        for data_list in ep_action_obs:
            for data in data_list:
                b_action_obs.append(data)
                
        b_pocket_obs: list[Data] = []
        for data_list in ep_pocket_obs:
            for data in data_list:
                b_pocket_obs.append(data)
        
        b_frag_actions = torch.cat(ep_actions)
        b_frag_logprobs = torch.cat(ep_logprobs)
        b_advantages = torch.cat(advantages)
        b_returns = torch.cat(returns)
        b_values = torch.cat(ep_values)
        b_masks = torch.cat(ep_masks)
        # b_entropies = torch.cat(ep_entropies)
        
        logging.info(b_returns)
    
    training_loop(agent=agent,
                b_action_obs=b_action_obs,
                b_pocket_obs=b_pocket_obs,
                b_frag_actions=b_frag_actions,
                b_frag_logprobs=b_frag_logprobs,
                b_advantages=b_advantages,
                b_returns=b_returns,
                b_values=b_values,
                b_masks=b_masks,
                n_epochs=n_epochs)
  
    # import pdb;pdb.set_trace()
  
    writer.add_scalar(f"train/mean_reward", returns[0].mean().item(), episode_i)

n_real_data = n_envs + 1

try:

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
        
        # train_i = []
        # current_i = 0
        # while len(train_i) < n_envs:
        #     train_i.append(current_i)
        #     current_i = (current_i + 1) % n_complexes
        
        seed_i = episode_i % n_complexes
        train_i = [seed_i] * n_envs
            
        if seed_i == 0 and n_real_data > 1:
            n_real_data -= 1
            
        episode(train_i,
                batch_env,
                # n_real_data=n_real_data
                )
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # import pdb;pdb.set_trace()
        
        # Save memory every memory_save_step
        memory_size = len(memory)
        current_memory_i = memory_size // memory_save_step
        if (current_memory_i > memory_i):
            with open(memory_path, 'wb') as f:
                pickle.dump(memory, f)
            memory_i = current_memory_i
        logging.info(f'Memory size: {memory_size}')
        
        # with torch.no_grad():
        #     episode(val_seed_idxs, val_batch_env, train=False)
        
        # if ((episode_i + 1) % 500 == 0):
        #     import pdb;pdb.set_trace()
        
        if ((episode_i + 1) % 500 == 0):
            save_path = f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt'
            torch.save(agent.state_dict(), f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt')
            
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
        
# except Exception as e:
#     print(e)
#     import pdb;pdb.set_trace()
#     print(e)
        
# agent = Agent(*args, **kwargs)
# agent.load_state_dict(torch.load(PATH))
# agent.eval()