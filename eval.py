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
from ymir.old.molecule_builder import potential_reactions
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
n_episodes = 9_000
batch_size = 16 # NN batch, input is Data, output are actions + predicted reward
n_envs = 64 # we will have protein envs in parallel
n_steps = 2 # number of maximum fragment growing
n_epochs = 5 # number of times we update the network per episode
lr = 1e-3
gamma = 0.99 # discount factor for rewards
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
complexes = []
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
protected_fragments = protected_fragments[:50]

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
state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir__1705709382_9000.pt')
agent.load_state_dict(state_dict)

agent.eval()

with torch.no_grad():

    complx_i = np.random.choice(n_complexes)
    complx = complexes[complx_i]

    next_info = batch_env.reset(complx)

    current_obs = batch_env.get_obs()
    current_masks = batch_env.get_valid_action_mask()
    x = Batch.from_data_list(data_list=current_obs)
    x.to(device)
    features = agent.extract_features(x)
    current_masks = current_masks.to(device)
    current_action, current_logprob, current_entropy = agent.get_action(features, 
                                                                        masks=current_masks)
    t = batch_env.step(actions=current_action)

    batch_env.save_state()
    