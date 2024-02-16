import time
import torch
import random
import numpy as np
import logging

from torch import nn
from tqdm import tqdm
from ymir.env import (FragmentBuilderEnv, 
                      BatchEnv, 
                      Action)
from ymir.params import (TORSION_SPACE_STEP,
                         FEATURES_DIM)
from ymir.fragment_library import FragmentLibrary
from ymir.data.structure import Complex
from torch.utils.tensorboard import SummaryWriter
from ymir.utils.fragment import get_fragments_from_mol
from ymir.policy import Agent
from torch.optim import Adam
from ymir.reward import DockingBatchRewards
from torch_geometric.data import Batch, Data
from ymir.molecule_builder import potential_reactions
from ymir.metrics.activity import GlideScore, VinaScore, VinaScorer

logging.basicConfig(filename='train.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filemode="w")

experiment_name = f"ymir__{int(time.time())}"
writer = SummaryWriter(f"logs/{experiment_name}")

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# 1 episode = grow fragments + update NN
n_episodes = 100_000
batch_size = 16 # NN batch, input is Data, output are actions + predicted reward
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

fragment_library = FragmentLibrary()
ligands = fragment_library.ligands
protected_fragments = fragment_library.protected_fragments

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)
    
assert len(ligands) == len(protein_paths)

logging.info('Loading complexes')

n_complexes = 1

not_working_protein = []
complexes: list[Complex] = []
for protein_path, ligand in zip(protein_paths[:n_complexes], ligands[:n_complexes]):
    try:
        complx = Complex(ligand, protein_path)
        fragments = get_fragments_from_mol(ligand)
        assert len(fragments) > 1, 'Ligand is not fragmentable'
    except Exception as e:
        logging.warning(f'Error on {protein_path}: {e}')
        not_working_protein.append(protein_path)
    else:
        complexes.append(complx)

torsion_space_step = TORSION_SPACE_STEP
torsion_values = np.arange(-180, 180, torsion_space_step)

# TO CHANGE/REMOVE
protected_fragments = protected_fragments[:500]

possible_actions: list[Action] = []
for fragment_i in range(len(protected_fragments)):
    for torsion_value in torsion_values:
        action= Action(fragment_i=fragment_i,
                        torsion_value=torsion_value)
        possible_actions.append(action)
action_dim = len(possible_actions)

logging.info('Loading valid action masks')
valid_action_masks: dict[int, list[bool]] = {}
for attach_label_1, d_potential_attach in potential_reactions.items():
    mask = [False for _ in range(action_dim)]
    
    # for i in range(action_dim):
    #     if i % attach_label_1 == 0: # fake mask
    #         mask[i] = True
    
    n_true = 0
    for act_i, action in enumerate(possible_actions):
        fragment_i = action.fragment_i
        fragment = protected_fragments[fragment_i]
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() == 0:
                attach_label = atom.GetIsotope()
                break
        # attach_points = fragment.get_attach_points()
        # assert len(attach_points) == 1
        # attach_label = list(attach_points.values())[0]
        if attach_label in d_potential_attach:
            mask[act_i] = True
            n_true += 1
        
            # if n_true == 100: # TO REMOVE
            #     break
            
    valid_action_masks[attach_label_1] = torch.tensor(mask, dtype=torch.bool)

logging.info(f'There are {len(protected_fragments)} fragments')

envs: list[FragmentBuilderEnv] = [FragmentBuilderEnv(protected_fragments=protected_fragments,
                                                    torsion_space_step=torsion_space_step,
                                                    max_episode_steps=n_steps,
                                                    valid_action_masks=valid_action_masks)
                                  for _ in range(n_envs)]
action_dim = envs[0].action_dim
batch_env = BatchEnv(envs)

def mask_fn(env: FragmentBuilderEnv) -> list[bool]:
    return env.get_valid_action_mask()

agent = Agent(action_dim, 
              features_dim=FEATURES_DIM)
# state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir__1705682497_900.pt')
# agent.load_state_dict(state_dict)

optimizer = Adam(agent.parameters(), lr=lr)

# envs need to be set with a complex
# reward need to be batched
# torch_geometric Batch for action/value prediction
# select obs (input of NN) that are not terminated

n_complexes = len(complexes)
logging.info(f'Training on {n_complexes} complexes')

logging.info('Start RL')

try:

    for episode_i in tqdm(range(n_episodes)):
        
        logging.debug(f'Episode i: {episode_i}')
        
        complx_i = np.random.choice(n_complexes)
        complx = complexes[complx_i]
        
        # scorer = GlideScore(glide_protein=complx.glide_protein)
        # vina_scorer = VinaScorer(vina_protein=complx.vina_protein)
        # vina_scorer.set_box_from_ligand(complx.ligand)
        # scorer = VinaScore(vina_scorer)
        # reward_function = DockingBatchRewards(scorer,
        #                                       pocket=complx.pocket)
        
        # current_obs, next_info = batch_env.reset(complx)
        next_info = batch_env.reset(complx)
        next_terminated = [False] * n_envs
        
        # each first dimension of list is the number of total steps
        obs: list[list[Data]] = []
        actions: list[torch.Tensor] = [] # (n_non_term_envs)
        logprobs: list[torch.Tensor] = [] # (n_non_term_envs)
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
                current_action, current_logprob, current_entropy = agent.get_action(features, 
                                                                                    masks=current_masks)
                current_value = agent.get_value(features)
                current_value = current_value.squeeze(dim=-1)
                
            actions.append(current_action.cpu())
            logprobs.append(current_logprob.cpu())
            values.append(current_value.cpu())
            
            # import pdb;pdb.set_trace()
            t = batch_env.step(actions=current_action)
            # next_obs, _, next_terminated, next_truncated, next_info = t
            reward, next_terminated, next_truncated, next_info = t
            
            rewards.append(torch.tensor(reward))
            
            step_i += 1
            
        if step_i == n_steps:
            assert all(next_truncated), 'All molecule generation should be stopped'
            
        batch_env.save_state()
            
        # mols = batch_env.get_mols()
        # rewards = reward_function.get_rewards(mols)
        # rewards = torch.randint(low=0,
        #                         high=10,
        #                         size=(n_envs,))
        # rewards = rewards.clip(min=-100)
        
        logging.info(rewards)
        
        with torch.no_grad():
            reversed_values = reversed(values)
            reversed_rewards = reversed(rewards)
            reversed_terminateds = reversed(terminateds)
            reversed_returns = [] # (n_non_term_envs, 1)
            reversed_advantages = [] # (n_non_term_envs, 1)
            # last_values = {}
            # last_returns = {env_i: 0 for env_i in range(n_envs)}
            last_obs = batch_env.get_obs(all_envs=True)
            last_batch = Batch.from_data_list(data_list=last_obs)
            last_batch = last_batch.to(device)
            last_features = agent.extract_features(x=last_batch)
            last_values = agent.get_value(features=last_features)
            lastgaelams = {env_i: 0 for env_i in range(n_envs)}
            z = zip(reversed_values, reversed_terminateds, reversed_rewards)
            for step_values, step_terminated, step_rewards in z:
                step_non_terminated = [not terminated for terminated in step_terminated]
                non_terminated_idxs = np.where(step_non_terminated)[0]
                
                try:
                    assert len(non_terminated_idxs) == step_values.shape[0]
                except:
                    import pdb;pdb.set_trace()
                
                # # update last value
                # for env_i, value in zip(non_terminated_idxs, step_values):
                #     # if not env_i in last_values:
                #     last_values[env_i] = value
                
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
        
        b_logprobs = torch.cat(logprobs)
        b_actions = torch.cat(actions)
        b_advantages = torch.cat(advantages)
        b_returns = torch.cat(returns)
        b_values = torch.cat(values)
        b_masks = torch.cat(masks)
        
        assert len(b_obs) == b_logprobs.size()[0]
        assert len(b_obs) == b_actions.size()[0]
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
                mb_advantages = b_advantages[minibatch_inds]
                mb_advantages = mb_advantages.to(device)
                
                # if mb_advantages.size()[0] > 1:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                mb_obs = [obs 
                        for i, obs in enumerate(b_obs) 
                        if i in minibatch_inds]
                mb_actions = b_actions[minibatch_inds]
                mb_masks = b_masks[minibatch_inds]
                
                mb_batch = Batch.from_data_list(mb_obs)
                mb_batch = mb_batch.to(device)
                mb_features = agent.extract_features(mb_batch)
                
                mb_masks = mb_masks.to(device)
                mb_actions = mb_actions.to(device)
                _, newlogproba, entropy = agent.get_action(features=mb_features,
                                                        masks=mb_masks,
                                                        actions=mb_actions,)
                
                mb_logprobs = b_logprobs[minibatch_inds]
                mb_logprobs = mb_logprobs.to(device)
                ratio = (newlogproba - mb_logprobs).exp()
                
                approx_kl = (mb_logprobs - newlogproba).mean()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 
                                                        min=1 - clip_coef, 
                                                        max=1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
                
                mb_returns = b_returns[minibatch_inds]
                mb_values = b_values[minibatch_inds]
                mb_returns = mb_returns.to(device)
                mb_values = mb_values.to(device)
                new_values = agent.get_value(features=mb_features)
                v_loss_unclipped = ((new_values - mb_returns) ** 2)
                v_clipped = mb_values + torch.clamp(new_values - mb_values, 
                                                    -clip_coef, 
                                                    clip_coef)
                v_loss_clipped = (v_clipped - mb_returns)**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(parameters=agent.parameters(), 
                                        clip_value=max_grad_value)
                optimizer.step()
                
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], episode_i)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode_i)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode_i)
        writer.add_scalar("losses/entropy", entropy.mean().item(), episode_i)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode_i)
        writer.add_scalar("losses/loss", loss.item(), episode_i)
        
        if ((episode_i + 1) % 10000 == 0):
            save_path = f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt'
            torch.save(agent.state_dict(), f'/home/bb596/hdd/ymir/models/{experiment_name}_{(episode_i + 1)}.pt')
            
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
        
    # import pdb;pdb.set_trace()
        
# agent = Agent(*args, **kwargs)
# agent.load_state_dict(torch.load(PATH))
# agent.eval()