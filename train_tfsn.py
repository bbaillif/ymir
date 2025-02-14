import time
import torch
import random
import numpy as np
import logging

from torch import nn
from tqdm import tqdm
from ymir.mace_tfsn_env import (FragmentBuilderEnv, 
                            BatchEnv)
from ymir.fragment_library import FragmentLibrary
from ymir.data.structure import Complex
from torch.utils.tensorboard import SummaryWriter
from ymir.utils.fragment import get_fragments_from_mol
from ymir.mace_tfsn_policy import Agent
from torch.optim import Adam
from torch_geometric.data import Batch, Data
from ymir.old.molecule_builder import potential_reactions
from e3nn import o3
from ymir.atomic_num_table import AtomicNumberTable
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean
from ymir.utils.spatial import rotate_conformer, translate_conformer
from rdkit import Chem
from datetime import datetime
from ymir.data import Fragment

logging.basicConfig(filename='train_tfsn.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode="w")

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# 1 episode = grow fragments + update NN
n_episodes = 100_000
n_envs = 128 # we will have protein envs in parallel
batch_size = n_envs # NN batch, input is Data, output are actions + predicted reward
n_steps = 10 # number of maximum fragment growing
n_epochs = 5 # number of times we update the network per episode
lr = 5e-4
gamma = 0.95 # discount factor for rewards
gae_lambda = 0.95 # lambda factor for GAE
device = torch.device('cuda')
clip_coef = 0.5
ent_coef = 0.05
vf_coef = 0.5
max_grad_value = 0.5
hidden_irreps = o3.Irreps('8x0e + 8x1o + 8x2e + 8x3o')

n_complexes = 10

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_tfsn_{timestamp}"

writer = SummaryWriter(f"logs/{experiment_name}")

fragment_library = FragmentLibrary()
ligands = fragment_library.ligands
protected_fragments = fragment_library.protected_fragments

ligands = [ligand 
           for ligand in ligands 
           if ligand.GetNumHeavyAtoms() < 50]

random.shuffle(ligands)

# Select only fragments with less than 3 attach points
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
                       if n <= 3]

# with Chem.SDWriter('selected_fragments.sdf') as sdwriter:
#     for fragment in protected_fragments:
#         fragment.unprotect()
#         sdwriter.write(fragment)

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

set_atomic_nums = set()
for fragment in protected_fragments:
    for atom in fragment.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        set_atomic_nums.add(atomic_num)
set_atomic_nums.add(53)

z_table = AtomicNumberTable(zs=list(set_atomic_nums))

# TO CHANGE/REMOVE
# n_fragments = 1000
# protected_fragments = protected_fragments[:n_fragments]

# Align the attach point ---> neighbor vector to the x axis: (0,0,0) ---> (1,0,0)
# Then translate such that the neighbor is (0,0,0)
for fragment in protected_fragments:
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0:
            attach_point = atom
            break
    neighbor = attach_point.GetNeighbors()[0]
    neighbor_id = neighbor.GetIdx()
    attach_id = attach_point.GetIdx()
    positions = fragment.GetConformer().GetPositions()
    # neighbor_attach = positions[[neighbor_id, attach_id]]
    # distance = euclidean(neighbor_attach[0], neighbor_attach[1])
    # x_axis_vector = np.array([[0,0,0], [distance,0,0]])
    neighbor_pos = positions[neighbor_id]
    attach_pos = positions[attach_id]
    attach_neighbor = neighbor_pos - attach_pos
    distance = euclidean(neighbor_pos, attach_pos)
    x_axis_vector = np.array([distance, 0, 0])
    # import pdb;pdb.set_trace()
    rotation, rssd = Rotation.align_vectors(a=x_axis_vector.reshape(-1, 3), b=attach_neighbor.reshape(-1, 3))
    rotate_conformer(conformer=fragment.GetConformer(),
                        rotation=rotation)
    
    positions = fragment.GetConformer().GetPositions()
    neighbor_pos = positions[neighbor_id]
    translation = -neighbor_pos
    translate_conformer(conformer=fragment.GetConformer(),
                        translation=translation)

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
                                                     z_table=z_table,
                                                    max_episode_steps=n_steps,
                                                    valid_action_masks=valid_action_masks)
                                  for _ in range(n_envs)]
assert action_dim == envs[0].action_dim
batch_env = BatchEnv(envs)

agent = Agent(protected_fragments=protected_fragments, 
              hidden_irreps=hidden_irreps)
state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_tfsn_14_02_2024_19_12_15_500.pt')
agent.load_state_dict(state_dict)

fragment_features = agent.extract_fragment_features()

optimizer = Adam(agent.parameters(), lr=lr)

n_complexes = len(complexes)
logging.info(f'Training on {n_complexes} complexes')

logging.info('Start RL')

try:

    for episode_i in tqdm(range(n_episodes)):
        
        logging.debug(f'Episode i: {episode_i}')
        
        complx_i = episode_i % n_complexes
        current_complexes = [complexes[complx_i]] * n_envs
        
        next_info = batch_env.reset(current_complexes)
        next_terminated = [False] * n_envs
        
        # each first dimension of list is the number of total steps
        obs: list[list[Data]] = []
        frag_actions: list[torch.Tensor] = [] # (n_non_term_envs)
        frag_logprobs: list[torch.Tensor] = [] # (n_non_term_envs)
        vector_actions: list[torch.Tensor] = [] # (n_non_term_envs)
        vector_logprobs: list[torch.Tensor] = [] # (n_non_term_envs)
        values: list[torch.Tensor] = [] # (n_non_term_envs)
        masks: list[torch.Tensor] = [] # (n_non_term_envs, action_dim)
        terminateds: list[list[bool]] = [] # (n_steps, n_envs)
        rewards: list[torch.Tensor] = [] # (n_non_term_envs)
        
        # code annealing LR ?
        
        # batch_env.save_state()
        # import pdb;pdb.set_trace()
        
        start_time = time.time()
        
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
                
                current_action = agent.get_action(features, 
                                                  fragment_features=fragment_features,
                                                    masks=current_masks)
                current_frag_actions = current_action.frag_i
                current_vector_actions = current_action.vector
                current_values = agent.get_value(features)
                current_values = current_values.squeeze(dim=-1)
                
            frag_actions.append(current_frag_actions.cpu())
            frag_logprobs.append(current_action.frag_logprob.cpu())
            vector_actions.append(current_vector_actions.cpu())
            vector_logprobs.append(current_action.vector_logprob.cpu())
            values.append(current_values.cpu())
            
            # only select the vector info of the selected fragment
            arange = torch.arange(current_vector_actions.shape[0])
            current_vector_actions = current_vector_actions[arange, current_frag_actions]
            
            # import pdb;pdb.set_trace()
            t = batch_env.step(frag_actions=current_frag_actions.cpu(),
                               vector_actions=current_vector_actions.cpu())
            # next_obs, _, next_terminated, next_truncated, next_info = t
            
            logging.info(current_frag_actions)
            logging.info(current_vector_actions)
            
            reward, next_terminated, next_truncated, next_info = t
            
            rewards.append(torch.tensor(reward))
            
            step_i += 1
            
        batch_env.save_state()
            
        if step_i == n_steps:
            assert all(next_truncated), 'All molecule generation should be stopped'
            
        logging.info(f'First loop time: {time.time() - start_time}')
        
        logging.info(rewards)
        
        start_time = time.time()
        
        # with torch.no_grad():
        reversed_values = reversed(values)
        reversed_rewards = reversed(rewards)
        reversed_terminateds = reversed(terminateds)
        reversed_returns = [] # (n_non_term_envs, 1)
        reversed_advantages = [] # (n_non_term_envs, 1)
        last_values = {env_i: 0 for env_i in range(n_envs)}
        lastgaelams = {env_i: 0 for env_i in range(n_envs)}
        z = zip(reversed_values, reversed_terminateds, reversed_rewards)
        for step_values, step_terminated, step_rewards in z:
            step_non_terminated = [not terminated for terminated in step_terminated]
            non_terminated_idxs = np.where(step_non_terminated)[0]
            
            try:
                assert len(non_terminated_idxs) == step_values.shape[0]
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

        logging.info(f'Return loop time: {time.time() - start_time}')

        returns = list(reversed(reversed_returns))
        advantages = list(reversed(reversed_advantages))
        
        b_obs: list[Data] = []
        for data_list in obs:
            for data in data_list:
                b_obs.append(data)
        
        b_frag_actions = torch.cat(frag_actions)
        b_frag_logprobs = torch.cat(frag_logprobs)
        b_vector_actions = torch.cat(vector_actions)
        b_vector_logprobs = torch.cat(vector_logprobs)
        b_advantages = torch.cat(advantages)
        b_returns = torch.cat(returns)
        b_values = torch.cat(values)
        b_masks = torch.cat(masks)
        
        n_obs = len(b_obs)
        logging.info(f'{n_obs} obs for epochs')
        
        inds = np.arange(n_obs)
        
        start_time = time.time()
        
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
                    
                    mb_obs = [b_obs[i] for i in minibatch_inds]
                    mb_frag_actions = b_frag_actions[minibatch_inds]
                    mb_vector_actions = b_vector_actions[minibatch_inds]
                    mb_masks = b_masks[minibatch_inds]
                    
                    x = Batch.from_data_list(mb_obs)
                    x = x.to(device)
                    features = agent.extract_features(x)
                    
                    current_masks = mb_masks.to(device)
                    current_frag_actions = mb_frag_actions.to(device)
                    current_vector_actions = mb_vector_actions.to(device)
                    current_action = agent.get_action(features=features,
                                                      fragment_features=fragment_features,
                                                        masks=current_masks,
                                                        frag_actions=current_frag_actions,
                                                        vector_actions=current_vector_actions)
                    
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
                    
                    # only select the vector info of the selected fragment
                    arange = torch.arange(current_vector_actions.shape[0]).to(device)
                    current_vector_logprobs = current_action.vector_logprob
                    current_vector_logprobs = current_vector_logprobs[arange, current_frag_actions]
                    
                    mb_vector_logprobs = b_vector_logprobs[minibatch_inds]
                    mb_vector_logprobs = mb_vector_logprobs.to(device)
                    mb_vector_logprobs = mb_vector_logprobs[arange, current_frag_actions]
                    
                    vector_ratio = (current_vector_logprobs - mb_vector_logprobs).exp()
                    
                    vector_approx_kl = (mb_vector_logprobs - current_vector_logprobs).mean()
                    
                    vector_pg_loss1 = -mb_advantages * vector_ratio
                    vector_pg_loss2 = -mb_advantages * torch.clamp(vector_ratio, 
                                                            min=1 - clip_coef, 
                                                            max=1 + clip_coef)
                    vector_pg_loss = torch.max(vector_pg_loss1, vector_pg_loss2).mean()
                    
                    mb_returns = b_returns[minibatch_inds]
                    mb_values = b_values[minibatch_inds]
                    mb_returns = mb_returns.to(device)
                    mb_values = mb_values.to(device)
                    current_values = agent.get_value(features=features)
                    v_loss_unclipped = ((current_values - mb_returns) ** 2)
                    v_clipped = mb_values + torch.clamp(current_values - mb_values, 
                                                        -clip_coef, 
                                                        clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                    
                    loss = frag_pg_loss + vector_pg_loss + (v_loss * vf_coef) 
                    # loss = loss - (ent_coef * frag_entropy_loss)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    nn.utils.clip_grad_value_(parameters=agent.parameters(), 
                                            clip_value=max_grad_value)
                    optimizer.step()
                    
                    # Reload fragment_features
                    fragment_features = agent.extract_fragment_features()
        
        logging.info(f'Second loop time: {time.time() - start_time}')
                
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], episode_i)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode_i)
        writer.add_scalar("losses/fragment_policy_loss", frag_pg_loss.item(), episode_i)
        writer.add_scalar("losses/fragment_entropy", frag_entropy_loss.mean().item(), episode_i)
        writer.add_scalar("losses/fragment_approx_kl", frag_approx_kl.item(), episode_i)
        writer.add_scalar("losses/vector_policy_loss", vector_pg_loss.item(), episode_i)
        writer.add_scalar("losses/vector_approx_kl", vector_approx_kl.item(), episode_i)
        writer.add_scalar("losses/loss", loss.item(), episode_i)
        
        flat_rewards = torch.cat(rewards)
        writer.add_scalar("reward/mean_reward", flat_rewards.mean(), episode_i)
        
        # import pdb;pdb.set_trace()
        
        if ((episode_i + 1) % 500 == 0):
            save_path = f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt'
            torch.save(agent.state_dict(), f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt')
            
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
        
    # import pdb;pdb.set_trace()
        
# agent = Agent(*args, **kwargs)
# agent.load_state_dict(torch.load(PATH))
# agent.eval()