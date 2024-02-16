import torch
import random
import numpy as np
import logging

from datetime import datetime
from torch import nn
from tqdm import tqdm
from ymir.env_vm import (FragmentBuilderEnv, 
                      BatchEnv)
from ymir.params import FEATURES_DIM
from ymir.fragment_library import FragmentLibrary
from ymir.data.structure import Complex
from torch.utils.tensorboard import SummaryWriter
from ymir.utils.fragment import get_fragments_from_mol
from ymir.policy_vm import Agent
from torch.optim import Adam
from ymir.reward import DockingBatchRewards
from torch_geometric.data import Batch, Data
from ymir.molecule_builder import potential_reactions
from ymir.metrics.activity import GlideScore, VinaScore, VinaScorer

logging.basicConfig(filename='train_vm.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode="w")

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_vm_{timestamp}"
writer = SummaryWriter(f"logs/{experiment_name}")

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# 1 episode = grow fragments + update NN
n_episodes = 100_000
batch_size = 8 # NN batch, input is Data, output are actions + predicted reward
n_envs = 64 # we will have protein envs in parallel
n_steps = 10 # number of maximum fragment growing
n_epochs = 5 # number of times we update the network per episode
lr = 1e-4
gamma = 0.99 # discount factor for rewards
gae_lambda = 0.95 # lambda factor for GAE
device = torch.device('cuda')
clip_coef = 0.1
ent_coef = 0.01
vf_coef = 0.5
max_grad_value = 1.0

# n_fragments = 500
n_complexes = 500

fragment_library = FragmentLibrary()
ligands = fragment_library.ligands
protected_fragments = fragment_library.protected_fragments

ligands = [ligand 
           for ligand in ligands 
           if ligand.GetNumHeavyAtoms() < 50]

random.shuffle(ligands)

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)
    
assert len(ligands) == len(protein_paths)

logging.info('Loading complexes')
not_working_protein = []
complexes: list[Complex] = []
for protein_path, ligand in zip(protein_paths[:n_complexes], ligands[:n_complexes]):
    try:
        complx = Complex(ligand, protein_path)
        fragments = get_fragments_from_mol(ligand)
        for fragment in fragments:
            attach_points = fragment.get_attach_points()
            if 7 in attach_points.values():
                raise Exception("double bond BRICS, we don't include these")
        assert len(fragments) > 1, 'Ligand is not fragmentable (or double bond reaction)'
    except Exception as e:
        logging.warning(f'Error on {protein_path}: {e}')
        not_working_protein.append(protein_path)
    else:
        complexes.append(complx)

# TO CHANGE/REMOVE
# protected_fragments = protected_fragments[:n_fragments]

action_dim = len(protected_fragments)

fragment_attach_labels = []
for act_i, fragment in enumerate(protected_fragments):
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0:
            attach_label = atom.GetIsotope()
            break
    fragment_attach_labels.append(attach_label)

logging.info('Loading valid action masks')
valid_action_masks: dict[int, list[bool]] = {}
for attach_label_1, d_potential_attach in potential_reactions.items():
    mask = [True 
            if attach_label in d_potential_attach 
            else False
            for attach_label in fragment_attach_labels]
             
    valid_action_masks[attach_label_1] = torch.tensor(mask, dtype=torch.bool)

logging.info(f'There are {len(protected_fragments)} fragments')

envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(protected_fragments=protected_fragments,
                                                    max_episode_steps=n_steps,
                                                    valid_action_masks=valid_action_masks)
                                  for _ in range(n_envs)]
action_dim = envs[0].action_dim
batch_env = BatchEnv(envs)

agent = Agent(action_dim, 
              features_dim=FEATURES_DIM)
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir__1705682497_900.pt')
# agent.load_state_dict(state_dict)

optimizer = Adam(agent.parameters(), lr=lr)

n_complexes = len(complexes)
logging.info(f'Training on {n_complexes} complexes')

logging.info('Start RL')

try:

    for episode_i in tqdm(range(n_episodes)):
        
        logging.debug(f'Episode i: {episode_i}')
        
        # complx_i = np.random.choice(n_complexes)
        complx_i = episode_i % n_complexes 
        complx = complexes[complx_i]
        
        # print(complx.pocket.mol.GetNumAtoms())
        # print(complx.protein_path)
        
        try:
        
            # current_obs, next_info = batch_env.reset(complx)
            next_info = batch_env.reset(complx)
            next_terminated = [False] * n_envs
            
            # each first dimension of list is the number of total steps
            obs: list[list[Data]] = []
            frag_actions: list[torch.Tensor] = [] # (n_non_term_envs)
            angle_actions: list[torch.Tensor] = [] # (n_non_term_envs)
            frag_logprobs: list[torch.Tensor] = [] # (n_non_term_envs)
            angle_logprobs: list[torch.Tensor] = [] # (n_non_term_envs)
            values: list[torch.Tensor] = [] # (n_non_term_envs)
            masks: list[torch.Tensor] = [] # (n_non_term_envs, action_dim)
            terminateds: list[list[bool]] = [] # (n_steps, n_envs)
            rewards: list[torch.Tensor] = [] # (n_non_term_envs)
            
            # code annealing LR ?
            
            step_i = 0
            while step_i < n_steps and not all(next_terminated):
                
                logging.debug(f'Step i: {episode_i}')
                
                current_info = next_info
                current_terminated = next_terminated
                terminateds.append(current_terminated)
                
                current_obs = batch_env.get_obs()
                obs.append(current_obs)
                
                current_masks = batch_env.get_valid_action_mask()
                masks.append(current_masks)
                
                with torch.no_grad():
                    x = Batch.from_data_list(data_list=current_obs)
                    x.to(device)
                    features = agent.extract_features(x)
                    current_masks = current_masks.to(device)
                    
                    # Fragment action
                    t_frag = agent.get_frag_actions(features, masks=current_masks)
                    current_frag_action, current_frag_logprob, current_entropy = t_frag
                    frag_actions.append(current_frag_action.cpu())
                    frag_logprobs.append(current_frag_logprob.cpu())
                    
                    # Angle action
                    t_angle = agent.get_angle_actions(features)
                    current_angle_action, current_angle_logprob = t_angle
                    angle_actions.append(current_angle_action.cpu())
                    angle_logprobs.append(current_angle_logprob.cpu())
                    
                    # Critic
                    current_value = agent.get_value(features)
                    current_value = current_value.squeeze(dim=-1)
                    values.append(current_value.cpu())
                    
                t = batch_env.step(frag_actions=current_frag_action,
                                angle_actions=current_angle_action)
                reward, next_terminated, next_truncated, next_info = t
                
                rewards.append(torch.tensor(reward))
                
                step_i += 1
                
            if step_i == n_steps:
                assert all(next_truncated), 'All molecule generation should be stopped'
                
            batch_env.save_state()
            
            logging.info(rewards)
            
            reversed_values = reversed(values)
            reversed_rewards = reversed(rewards)
            reversed_terminateds = reversed(terminateds)
            reversed_returns = [] # (n_non_term_envs, 1)
            reversed_advantages = [] # (n_non_term_envs, 1)
            # last_values = {}
            # last_returns = {env_i: 0 for env_i in range(n_envs)}
            # last_obs = batch_env.get_obs(all_envs=True)
            # last_batch = Batch.from_data_list(data_list=last_obs)
            # last_batch = last_batch.to(device)
            # last_features = agent.extract_features(x=last_batch)
            # last_values = agent.get_value(features=last_features)
            last_values = {env_i: 0 for env_i in range(n_envs)}
            lastgaelams = {env_i: 0 for env_i in range(n_envs)}
            z = zip(reversed_values, reversed_terminateds, reversed_rewards)
            for step_values, step_terminated, step_rewards in z:
                step_non_terminated = [not terminated for terminated in step_terminated]
                non_terminated_idxs = np.where(step_non_terminated)[0]
                
                assert len(non_terminated_idxs) == step_values.shape[0]
                
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
            
            b_obs: list[Data] = []
            for data_list in obs:
                for data in data_list:
                    b_obs.append(data.to(device))
            
            b_frag_actions = torch.cat(frag_actions)
            b_angle_actions = torch.cat(angle_actions)
            b_frag_logprobs = torch.cat(frag_logprobs)
            b_angle_logprobs = torch.cat(angle_logprobs)
            b_advantages = torch.cat(advantages)
            b_returns = torch.cat(returns)
            b_values = torch.cat(values)
            b_masks = torch.cat(masks)
            
            assert len(b_obs) == b_frag_actions.size()[0]
            assert len(b_obs) == b_angle_actions.size()[0]
            assert len(b_obs) == b_frag_logprobs.size()[0]
            assert len(b_obs) == b_angle_logprobs.size()[0]
            assert len(b_obs) == b_advantages.size()[0]
            assert len(b_obs) == b_returns.size()[0]
            assert len(b_obs) == b_values.size()[0]
            assert len(b_obs) == b_masks.size()[0]
            
            n_obs = len(b_obs)
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
                                        
                        mb_obs = [obs 
                                for i, obs in enumerate(b_obs) 
                                if i in minibatch_inds]
                        
                        x = Batch.from_data_list(mb_obs)
                        x = x.to(device)
                        features = agent.extract_features(x)
                        
                        # Fragment loss
                        current_frag_action = b_frag_actions[minibatch_inds]
                        current_frag_action = current_frag_action.to(device)
                        current_masks = b_masks[minibatch_inds]
                        current_masks = current_masks.to(device)
                        t_frag = agent.get_frag_actions(features=features,
                                                        masks=current_masks,
                                                        frag_actions=current_frag_action)
                        _, current_frag_logprob, current_entropy = t_frag
                        
                        mb_frag_logprobs = b_frag_logprobs[minibatch_inds]
                        mb_frag_logprobs = mb_frag_logprobs.to(device)
                        frag_ratio = (current_frag_logprob - mb_frag_logprobs).exp()
                        
                        frag_approx_kl = (mb_frag_logprobs - current_frag_logprob).mean()
                        
                        frag_pg_loss1 = -mb_advantages * frag_ratio
                        frag_pg_loss2 = -mb_advantages * torch.clamp(frag_ratio, 
                                                                min=1 - clip_coef, 
                                                                max=1 + clip_coef)
                        frag_pg_loss = torch.max(frag_pg_loss1, frag_pg_loss2).mean()
                        entropy_loss = current_entropy.mean()
                        
                        # Angle loss
                        current_angle_action = b_angle_actions[minibatch_inds]
                        current_angle_action = current_angle_action.to(device)
                        _, current_angle_logprob = agent.get_angle_actions(features=features,
                                                                        angle_actions=current_angle_action)
                        
                        mb_angle_logprobs = b_angle_logprobs[minibatch_inds]
                        mb_angle_logprobs = mb_angle_logprobs.to(device)
                        
                        # select angle_action for frag_action
                        mb_angle_logprobs = mb_angle_logprobs[torch.arange(n_sample), current_frag_action]
                        current_angle_logprob = current_angle_logprob[torch.arange(n_sample), current_frag_action]
                        
                        angle_ratio = (current_angle_logprob - mb_angle_logprobs).exp()
                        
                        angle_approx_kl = (mb_angle_logprobs - current_angle_logprob).mean()
                        
                        angle_pg_loss1 = -mb_advantages * angle_ratio
                        angle_pg_loss2 = -mb_advantages * torch.clamp(angle_ratio, 
                                                                min=1 - clip_coef, 
                                                                max=1 + clip_coef)
                        angle_pg_loss = torch.max(angle_pg_loss1, angle_pg_loss2).mean()
                        
                        # Critic loss
                        mb_returns = b_returns[minibatch_inds]
                        mb_values = b_values[minibatch_inds]
                        mb_returns = mb_returns.to(device)
                        mb_values = mb_values.to(device)
                        current_value = agent.get_value(features=features)
                        v_loss_unclipped = ((current_value - mb_returns) ** 2)
                        v_clipped = mb_values + torch.clamp(current_value - mb_values, 
                                                            -clip_coef, 
                                                            clip_coef)
                        v_loss_clipped = (v_clipped - mb_returns)**2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                        
                        loss = frag_pg_loss + angle_pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                        
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_value_(parameters=agent.parameters(), 
                                                clip_value=max_grad_value)
                        optimizer.step()
                    
            # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], episode_i)
            writer.add_scalar("losses/value_loss", v_loss.item(), episode_i)
            writer.add_scalar("losses/fragment_policy_loss", frag_pg_loss.item(), episode_i)
            writer.add_scalar("losses/angle_policy_loss", angle_pg_loss.item(), episode_i)
            writer.add_scalar("losses/frag_entropy", current_entropy.mean().item(), episode_i)
            writer.add_scalar("losses/frag_approx_kl", frag_approx_kl.item(), episode_i)
            writer.add_scalar("losses/angle_approx_kl", angle_approx_kl.item(), episode_i)
            writer.add_scalar("losses/loss", loss.item(), episode_i)
            
            flat_rewards = torch.cat(rewards)
            writer.add_scalar("reward/mean_reward", flat_rewards.mean(), episode_i)
            
            if ((episode_i + 1) % 500 == 0):
                save_path = f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt'
                torch.save(agent.state_dict(), f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt')
                
        except Exception as e:
            logging.warning(f'Something went wrong: {e}')
            
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
except Exception as e:
    print(e)
    import pdb;pdb.set_trace()
    