import logging
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Mol
from torch import nn
from torch_geometric.data import Data, Batch
from ymir.data import Fragment, Complex
from ymir.data.structure import GlideProtein
from ymir.utils.fragment import get_neighbor_id_for_atom_id, find_broken_bond, find_broken_bond_for_attach, merge_mappings
from ymir.molecule_builder import potential_reactions
from ymir.policy import Agent, Action
from ymir.env import BatchEnv
from ymir.distribution import CategoricalMasked
from torch.utils.tensorboard import SummaryWriter
from ymir.molecule_builder import add_fragment_to_seed


def get_frag_i(fragment: Fragment, 
               atom_id_to_keep: int,
               smiles_combos: list[tuple[str, str]]):
    frag_copy = Fragment.from_fragment(fragment)
    up_smiles = Chem.MolToSmiles(frag_copy.mol)
    # if up_smiles == '[1*]C([6*])=O':
    #     import pdb;pdb.set_trace()
    frag_copy.protect([atom_id_to_keep])
    p_smiles = Chem.MolToSmiles(frag_copy.mol)
    combo = (p_smiles, up_smiles)
    idxs = [i for i, smiles_combo in enumerate(smiles_combos) if smiles_combo == combo]
    if len(idxs) != 1:
        import pdb;pdb.set_trace()
    try:
        frag_i = smiles_combos.index(combo)
    except Exception as e:
        print(str(e))
        import pdb;pdb.set_trace()
    
    return frag_i


# def get_path(ligand: Mol,
#              fragments: Fragment,
#              initial_frag_i: int,
#              frag_idxs: list[int],
#              frags_mol_atom_mapping: list[list[int]],
#              smiles_combos: list[tuple[str, str]]):
#     placed_fragments_i = [initial_frag_i]
#     # master_frag_i = get_master_frag_i(current_fragment)
#     action_sequence = []
    
#     while len(action_sequence) != (len(fragments) - 1):
#         current_frags_i = list(placed_fragments_i)
#         for frag_i in current_frags_i:
#             current_fragment = fragments[frag_i]
#             for other_frag_i in frag_idxs:
#                 other_fragment = fragments[other_frag_i]
#                 if other_frag_i not in placed_fragments_i:
#                     broken_bond = find_broken_bond(ligand, current_fragment, frag_i, 
#                                                 other_fragment, other_frag_i, frags_mol_atom_mapping)
#                     if broken_bond:
#                         neigh_id1, attach_point, neigh_id2, attach_point2 = broken_bond
#                         attach_position = current_fragment.mol.GetConformer().GetPositions()[attach_point]
#                         master_frag_i = get_frag_i(other_fragment, 
#                                                    atom_id_to_keep=attach_point2,
#                                                    smiles_combos=smiles_combos)
#                         # if master_frag_i == 4 and ligand_name == '6t1j_ligand':
#                         #     import pdb;pdb.set_trace()
#                         action = (attach_position, master_frag_i)
#                         action_sequence.append(action)
#                         placed_fragments_i.append(other_frag_i)
                        
#     return action_sequence


def get_path(ligand: Mol,
             fragments: list[Fragment],
             initial_frag_i: int,
             frag_idxs: list[int],
             frags_mol_atom_mapping: list[list[int]],
             smiles_combos: list[tuple[str, str]]):
    placed_fragments_i = [initial_frag_i]
    seed = Fragment.from_fragment(fragments[initial_frag_i])
    seed_mapping = frags_mol_atom_mapping[initial_frag_i]
    # master_frag_i = get_master_frag_i(current_fragment)
    action_sequence = []
    
    while len(action_sequence) != (len(fragments) - 1):
        aps = seed.get_attach_points()
        attach_point, label = list(aps.items())[0] # only the first one to avoid randomness
        for other_frag_i in frag_idxs:
            if other_frag_i not in placed_fragments_i:
                other_fragment = Fragment.from_fragment(fragments[other_frag_i])
                frag_mapping = frags_mol_atom_mapping[other_frag_i]
                broken_bond = find_broken_bond_for_attach(attach_point, 
                                                          label, 
                                                          ligand, 
                                                          seed, 
                                                          other_fragment, 
                                                          seed_mapping, 
                                                          frag_mapping)
                # import pdb;pdb.set_trace()
                if broken_bond:
                    neigh_id1, attach_point, neigh_id2, attach_point2 = broken_bond
                    # attach_position = seed.mol.GetConformer().GetPositions()[attach_point]
                    master_frag_i = get_frag_i(other_fragment, 
                                                atom_id_to_keep=attach_point2,
                                                smiles_combos=smiles_combos)
                    # if master_frag_i == 4 and ligand_name == '6t1j_ligand':
                    #     import pdb;pdb.set_trace()
                    # action = (attach_position, master_frag_i)
                    action_sequence.append(master_frag_i)
                    placed_fragments_i.append(other_frag_i)
                    seed.protect(atom_ids_to_keep=[attach_point])
                    other_fragment.protect(atom_ids_to_keep=[attach_point2])
                    product, seed_to_product_mapping, fragment_to_product_mapping = add_fragment_to_seed(seed, other_fragment)
                    new_mapping = merge_mappings(seed_mapping,
                                                    frag_mapping,
                                                    seed_to_product_mapping,
                                                    fragment_to_product_mapping,
                                                    attach_point,
                                                    attach_point2)
                    seed = product
                    seed_mapping = new_mapping
                    break # refresh fragment list and first attach point
                        
    return action_sequence


def get_paths(ligand: Mol,
              fragments: list[Fragment],
              protein_path: str,
              lp_pdbbind_subsets: dict[str, str],
              glide_not_working_pdb_ids: list[str],
              frags_mol_atom_mapping,
              smiles_combos):
    seeds = []
    generation_sequences = []
    complexes = []
    ligand_name = ligand.GetProp('_Name')
    pdb_id = ligand_name.split('_')[0]
    if (pdb_id in lp_pdbbind_subsets) and (not pdb_id in glide_not_working_pdb_ids):
        subset = lp_pdbbind_subsets[pdb_id]
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
                
                frag_idxs = list(range(len(fragments)))
                for initial_frag_i in frag_idxs:
                    
                    action_sequence = get_path(ligand, 
                                               fragments, 
                                               initial_frag_i, 
                                               frag_idxs, 
                                               frags_mol_atom_mapping,
                                               smiles_combos)
                    
                    seeds.append(fragments[initial_frag_i])
                    generation_sequences.append(action_sequence)
                    complexes.append(complx)
                    
    return seeds, generation_sequences, complexes
    
    
def reinforce_episode(agent: Agent,
                        episode_i: int,
                        seed_idxs: list[int], 
                        seeds: list[Fragment],
                        complexes: list[Complex],
                        initial_scores: list[float],
                        native_scores: list[float],
                        generation_sequences: list[list[tuple[tuple[float], int]]],
                        batch_env: BatchEnv,
                        n_max_steps: int,
                        n_real_data: int,
                        device: torch.device,
                        gamma: float,
                        writer: SummaryWriter,
                        optimizer: torch.optim.Optimizer,
                        ent_coef,
                        use_entropy_loss,
                        pocket_feature_type):
    
    current_seeds = [seeds[seed_i] for seed_i in seed_idxs]
    current_complexes = [complexes[seed_i] for seed_i in seed_idxs]
    current_initial_scores = [initial_scores[seed_i] for seed_i in seed_idxs]
    current_native_scores = [native_scores[seed_i] for seed_i in seed_idxs]
    
    if n_real_data > 0:
        real_data_idxs = np.random.choice(range(len(seed_idxs)), n_real_data, replace=False)
    else:
        real_data_idxs = []
    current_generation_paths = [generation_sequences[seed_i] for seed_i in seed_idxs]
    
    next_info = batch_env.reset(current_complexes,
                                current_seeds,
                                current_initial_scores,
                                current_native_scores,
                                seed_idxs,
                                real_data_idxs,
                                current_generation_paths)
    n_envs = len(batch_env.envs)
    next_terminated = [False] * n_envs
    
    step_i = 0
    ep_masks = []
    ep_logprobs = []
    ep_rewards = []
    ep_entropies = []
    ep_terminateds: list[list[bool]] = [] # (n_steps, n_envs)
    while step_i < n_max_steps and not all(next_terminated):
        
        current_terminated = next_terminated
        current_obs = batch_env.get_obs()
        
        current_masks = batch_env.get_valid_action_mask()
        
        if pocket_feature_type == 'soap':
            # input = torch.stack([torch.tensor(obs, dtype=torch.float) for obs in current_obs])
            input = torch.tensor(current_obs, dtype=torch.float)
            input = input.reshape(-1, batch_env.pocket_feature_dim)
            input = input.to(device)
            features = agent.extract_features(input)
        
        else:
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
        current_frag_actions = current_action.frag_i
        current_frag_logprobs = current_action.frag_logprob
        
        try:
            if n_real_data > 0:
                for real_data_i in real_data_idxs:
                    if real_data_i in batch_env.ongoing_env_idxs:
                        ongoing_i = batch_env.ongoing_env_idxs.index(real_data_i)
                        master_frag_i = current_generation_paths[real_data_i][step_i]
                        current_frag_actions[ongoing_i] = master_frag_i
                        
                logprob = current_policy.log_prob(current_frag_actions)
                current_frag_logprobs = logprob  
            
        except Exception as e:
            print(str(e))
            import pdb;pdb.set_trace()
            
        current_frag_actions = current_action.frag_i.cpu()  
        
        t = batch_env.step(frag_actions=current_frag_actions)
        
        logging.info(f'Actions: {current_frag_actions}')
        logging.info(f"Probs: {current_frag_logprobs.exp()}")
        
        step_rewards, next_terminated, next_truncated = t
        
        ep_logprobs.append(current_frag_logprobs)
        ep_rewards.append(step_rewards)
        ep_entropies.append(current_action.frag_entropy)
        ep_terminateds.append(current_terminated)
        ep_masks.append(current_masks)
        
        step_i += 1
        
    batch_env.save_state()
    scores = np.array([min(env.current_score, 0) for env in batch_env.envs])
    writer.add_scalar(f"train/mean_score", scores.mean(), episode_i)
    writer.add_histogram(f"train/scores", scores, episode_i)
        
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
    all_masks = torch.cat(ep_masks)
    
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-5)
    
    logging.info(f'Returns: {all_returns}')
    
    n_possible_fragments = all_masks.sum(dim=-1)
    policy_loss = -all_logprobs * all_returns
    if use_entropy_loss:
        entropy_loss = - all_entropies / n_possible_fragments.log() 
        loss = policy_loss + entropy_loss * ent_coef
    else:
        loss = policy_loss
        
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar("train/policy_loss", policy_loss.mean().item(), episode_i)
    writer.add_scalar("train/entropy", all_entropies.mean().item(), episode_i)
    writer.add_scalar("train/loss", loss.item(), episode_i)
    writer.add_scalar(f"train/mean_reward", returns[0].mean().item(), episode_i)
    writer.add_scalar(f'params/ent_coef', ent_coef, episode_i)
    
    logging.info(f'Entropies: {all_entropies}')
    logging.info(f'Nfrags: {n_possible_fragments}')
    logging.info(f'log(Nfrags): {n_possible_fragments.log()}')
    logging.info(f'Entropy losses: {entropy_loss}')
    
    
    
def ppo_episode(agent: Agent,
            episode_i: int,
            seed_idxs: list[int], 
            seeds: list[Fragment],
            complexes: list[Complex],
            initial_scores: list[float],
            generation_sequences: list[list[tuple[tuple[float], int]]],
            batch_env: BatchEnv,
            n_max_steps: int,
            n_real_data: int,
            device: torch.device,
            gamma: float,
            gae_lambda: float,
            n_epochs: int,
            writer: SummaryWriter,
            optimizer: torch.optim.Optimizer,
            max_grad_value,
            vf_coef,
            ent_coef,
            batch_size,
            clip_coef,
            use_entropy_loss,
            ):
    
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
    n_envs = len(batch_env.envs)
    next_terminated = [False] * n_envs
    
    with torch.no_grad():
        ep_logprobs = []
        ep_rewards = []
        ep_obs = []
        ep_actions = []
        ep_masks = []
        ep_values = []
        ep_terminateds: list[list[bool]] = [] # (n_steps, n_envs)
        termination_step = [n_max_steps] * n_envs
        
        step_i = 0
        while step_i < n_max_steps and not all(next_terminated):
            
            current_terminated = next_terminated
            current_obs = batch_env.get_obs()
            ep_obs.append(current_obs)
            
            current_masks = batch_env.get_valid_action_mask()
            ep_masks.append(current_masks)
            
            batch = Batch.from_data_list(current_obs)
            batch = batch.to(device)
            
            features = agent.extract_features(batch)
            
            # input = torch.stack([torch.tensor(obs, dtype=torch.float) for obs in current_obs])
            # input = input.to(device)
            # features = agent.extract_features(input)
            
            # if current_masks.size()[0] != features.size()[0] :
            #     import pdb;pdb.set_trace()
            
            current_masks = current_masks.to(device)
            current_policy: CategoricalMasked = agent.get_policy(features,
                                                                #  batch=None,
                                                                 batch=batch.batch,
                                                                masks=current_masks)
            current_action: Action = agent.get_action(current_policy)
            current_frag_actions = current_action.frag_i
            current_frag_logprobs = current_action.frag_logprob
            
            try:
                if n_real_data > 0:
                    for real_data_i in real_data_idxs:
                        if real_data_i in batch_env.ongoing_env_idxs:
                            ongoing_i = batch_env.ongoing_env_idxs.index(real_data_i)
                            master_frag_i = current_generation_paths[real_data_i][step_i]
                            current_frag_actions[ongoing_i] = master_frag_i
                            
                    logprob = current_policy.log_prob(current_frag_actions)
                    current_frag_logprobs = logprob.cpu()
                    
            except Exception as e:
                print(str(e))
                import pdb;pdb.set_trace()
            
            current_frag_actions = current_frag_actions.cpu()
            current_frag_logprobs = current_frag_logprobs.cpu()
        
            ep_actions.append(current_frag_actions)
            
            t = batch_env.step(frag_actions=current_frag_actions)
            
            logging.info(current_frag_actions)
            logging.info(current_frag_logprobs.exp())
            
            step_rewards, next_terminated, next_truncated = t
            
            for env_i, (prev_term, new_term) in enumerate(zip(current_terminated, next_terminated)):
                if prev_term == False and new_term == True:
                    termination_step[env_i] = step_i
            
            ep_logprobs.append(current_frag_logprobs)
            ep_rewards.append(step_rewards)
            ep_terminateds.append(current_terminated)
            
            current_values = agent.get_value(features, 
                                            #  batch=None,
                                             batch=batch.batch
                                             )
            ep_values.append(current_values.cpu())
            
            step_i += 1
            
        assert all(next_terminated)
            
        # if SCORING_FUNCTION == 'glide':
        #     final_rewards = batch_env.get_rewards_glide(next_truncated, 
        #                                                 mininplace=True
        #                                                 )
        # else:
        #     final_rewards = batch_env.get_rewards(next_truncated)
        batch_env.save_state()
        scores = np.array([min(env.current_score, 0) for env in batch_env.envs])
        writer.add_scalar(f"train/mean_score", scores.mean(), episode_i)
        writer.add_histogram(f"train/scores", scores, episode_i)
        
        # ep_rewards = []
        # n_steps_in_episode = step_i
        # for step_i in range(n_steps_in_episode):
        #     step_rewards = []
        #     for env_i in range(n_envs):
        #         if step_i < termination_step[env_i]:
        #             step_rewards.append(0)
        #         elif step_i == termination_step[env_i]:
        #             step_rewards.append(final_rewards[env_i])
        #     ep_rewards.append(step_rewards)
            
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
            
        advantages = list(reversed(reversed_advantages))
        returns = list(reversed(reversed_returns))
        
        b_obs: list[Data] = []
        for data_list in ep_obs:
            for data in data_list:
                b_obs.append(data)
        
        b_frag_actions = torch.cat(ep_actions)
        b_frag_logprobs = torch.cat(ep_logprobs)
        b_advantages = torch.cat(advantages)
        b_returns = torch.cat(returns)
        b_values = torch.cat(ep_values)
        b_masks = torch.cat(ep_masks)
        
        logging.info(b_returns.squeeze())
    
    training_loop(agent=agent,
                    b_obs=b_obs,
                    b_frag_actions=b_frag_actions,
                    b_frag_logprobs=b_frag_logprobs,
                    b_advantages=b_advantages,
                    b_returns=b_returns,
                    b_values=b_values,
                    b_masks=b_masks,
                    n_epochs=n_epochs,
                    clip_coef=clip_coef,
                    use_entropy_loss=use_entropy_loss,
                    optimizer=optimizer,
                    episode_i=episode_i,
                    writer=writer,
                    max_grad_value=max_grad_value,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    batch_size=batch_size)
  
    # import pdb;pdb.set_trace()
  
    writer.add_scalar(f"train/mean_reward", returns[0].mean().item(), episode_i)
    
def training_loop(agent: Agent,
                  b_obs: list[Data],
                    b_frag_actions,
                    b_frag_logprobs,
                    b_advantages,
                    b_returns,
                    b_values,
                    b_masks,
                    batch_size,
                    clip_coef,
                    n_epochs,
                    use_entropy_loss,
                    optimizer,
                    episode_i,
                    writer: SummaryWriter,
                    max_grad_value,
                    vf_coef,
                    ent_coef,
                    ):
    n_obs = len(b_obs)
    logging.info(f'{n_obs} obs for epochs')
    device = agent.device
    
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
                
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-5)
                
                mb_obs = [b_obs[i] for i in minibatch_inds]
                mb_frag_actions = b_frag_actions[minibatch_inds]
                mb_masks = b_masks[minibatch_inds]
                
                batch = Batch.from_data_list(mb_obs)
                batch = batch.to(device)
                
                features = agent.extract_features(batch)
                
                # input = torch.stack([torch.tensor(obs, dtype=torch.float) for obs in mb_obs])
                # input = input.to(device)
                # features = agent.extract_features(input)
                
                current_masks = mb_masks.to(device)
                current_frag_actions = mb_frag_actions.to(device)
                current_policy = agent.get_policy(features=features,
                                                #   batch=None,
                                                    batch=batch.batch,
                                                    masks=current_masks,)
                current_action = agent.get_action(current_policy,
                                                    frag_actions=current_frag_actions)
                
                current_frag_logprobs = current_action.frag_logprob
                current_frag_entropy = current_action.frag_entropy
                
                mb_frag_logprobs = b_frag_logprobs[minibatch_inds]
                mb_frag_logprobs = mb_frag_logprobs.to(device)
                frag_ratio = (current_frag_logprobs - mb_frag_logprobs).exp()
                
                frag_approx_kl = (mb_frag_logprobs - current_frag_logprobs).mean()
                
                frag_pg_loss1 = -mb_advantages * frag_ratio.reshape(-1, 1)
                frag_pg_loss2 = -mb_advantages * torch.clamp(frag_ratio, 
                                                        min=1 - clip_coef, 
                                                        max=1 + clip_coef)
                frag_pg_loss = torch.max(frag_pg_loss1, frag_pg_loss2).mean()
                frag_entropy_loss = current_frag_entropy.mean()
                
                mb_returns = b_returns[minibatch_inds]
                mb_values = b_values[minibatch_inds]
                mb_returns = mb_returns.to(device)
                mb_values = mb_values.to(device)
                current_values = agent.get_value(features=features, 
                                                #  batch=None,
                                                 batch=batch.batch)
                v_loss_unclipped = ((current_values - mb_returns) ** 2)
                v_clipped = mb_values + torch.clamp(current_values - mb_values, 
                                                    -clip_coef, 
                                                    clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = v_loss_max.mean()
                
                loss = frag_pg_loss + (v_loss * vf_coef) 
                if torch.isnan(loss).any():
                    import pdb;pdb.set_trace()
                if use_entropy_loss:
                    loss = loss - (ent_coef * frag_entropy_loss)
                
                # if episode_i == 10 and epoch_i == 4:
                #     import pdb;pdb.set_trace()
                
                optimizer.zero_grad()
                loss.backward()
                
                try:
                    writer.add_histogram("gradients/actor",
                                torch.cat([p.grad.view(-1) for p in agent.actor.parameters()]), 
                                global_step=episode_i)
                    writer.add_histogram("gradients/critic",
                                torch.cat([p.grad.view(-1) for p in agent.critic.parameters()]), 
                                global_step=episode_i)
                except Exception as e:
                    print(str(e))
                    import pdb;pdb.set_trace()
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