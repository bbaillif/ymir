import time
import torch
import random
import numpy as np
import logging
import pandas as pd
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

from ymir.env import (FragmentBuilderEnv, 
                            BatchEnv)
from ymir.policy import Agent, Action
from ymir.data import Fragment
from ymir.params import (EMBED_HYDROGENS, 
                         HIDDEN_IRREPS,
                         SEED)
from ymir.metrics.activity import VinaScore, VinaScorer
import seaborn as sns
import matplotlib.pyplot as plt
from ymir.metrics.activity.vina_cli import VinaCLI
from rdkit.Chem.Descriptors import MolWt
from scipy.stats import spearmanr

logging.basicConfig(filename='evaluate.log', 
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
n_envs = 24 # we will have protein envs in parallel
batch_size = min(n_envs, 32) # NN batch, input is Data, output are actions + predicted reward
n_steps = 10 # number of maximum fragment growing
n_epochs = 5 # number of times we update the network per episode
lr = 1e-4
gamma = 0.95 # discount factor for rewards
gae_lambda = 0.95 # lambda factor for GAE
device = torch.device('cuda')
clip_coef = 0.5
ent_coef = 0.05
vf_coef = 0.5
max_grad_value = 0.5

n_complexes = 20
use_entropy_loss = False

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_v1_{timestamp}"

writer = SummaryWriter(f"logs/{experiment_name}")
# wandb.login()
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="ymir_v1",
#     # Track hyperparameters and run metadata
#     config={
#         "n_complexes": n_complexes,
#         "use_entropy_loss": False,
#         "lr": 5e-4,
#     },
# )

removeHs = not EMBED_HYDROGENS
fragment_library = FragmentLibrary(removeHs=removeHs)
ligands = fragment_library.ligands
protected_fragments = fragment_library.protected_fragments

ligands = [ligand 
           for ligand in ligands 
           if ligand.GetNumHeavyAtoms() < 50]

random.shuffle(ligands)

z_list = [0, 6, 7, 8, 16, 17]
if EMBED_HYDROGENS:
    z_list.append(1)
z_table = AtomicNumberTable(zs=z_list)
# Remove ligands having at least one heavy atom not in list
ligands = select_mol_with_symbols(ligands,
                                  z_list)

# Remove fragment having at least one heavy atom not in list
protected_fragments = select_mol_with_symbols(protected_fragments,
                                              z_list)

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
                       if n == 1]
           
# TO CHANGE/REMOVE
random.shuffle(protected_fragments)
# n_fragments = 100
# protected_fragments = protected_fragments[:n_fragments]
protected_fragments_smiles = [Chem.MolToSmiles(frag) for frag in protected_fragments]

# Get test set
fragment_library = FragmentLibrary(removeHs=removeHs, subset='general')
ligands = fragment_library.ligands
random.shuffle(ligands)

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = fragment_library.pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)
    
assert len(ligands) == len(protein_paths)

logging.info('Loading complexes')

# Get complexes with a correct Vina setup, and having a one-attach-point fragment in our list
not_working_protein = []
complexes: list[Complex] = []
seed_to_complex: list[int] = []
all_seeds: list[ConstructionSeed] = []
complex_counter = 0
for protein_path, ligand in tqdm(zip(protein_paths, ligands), total=len(protein_paths)):
    
    seeds = get_seeds(ligand)
    correct_seeds = []
    for seed in seeds:
        construct, removed_fragment = seed
        attach_points = construct.get_attach_points()
        assert len(attach_points) == 1
        if 7 in attach_points.values():
            continue
        
        frag_smiles = Chem.MolToSmiles(removed_fragment)
        if frag_smiles in protected_fragments_smiles:
            correct_seeds.append(seed)
    
    if len(correct_seeds) > 0:
        try:
            complx = Complex(ligand, protein_path)
            # vina_scorer = VinaScorer(complx.vina_protein) # Generate the Vina protein file
            vina_protein = complx.vina_protein
            pocket = complx.pocket # Detect the short pocket situations
            
        except Exception as e:
            logging.warning(f'Error on {protein_path}: {e}')
            not_working_protein.append(protein_path)
            
        else:
            complexes.append(complx)
            for seed in correct_seeds:
                all_seeds.append(seed)
                seed_to_complex.append(complex_counter)
            complex_counter += 1
            
            # TO REMOVE IN FINAL
            if complex_counter == n_complexes:
                break
            
n_seeds = len(all_seeds)

center_fragments(protected_fragments)
final_fragments = protected_fragments

valid_action_masks = get_masks(final_fragments)

logging.info(f'There are {len(final_fragments)} fragments')

agent = Agent(protected_fragments=final_fragments,
              atomic_num_table=z_table)
state_dict = torch.load('/home/bb596/hdd/ymir/models/ymir_v1_04_03_2024_01_21_47_22500.pt')

agent.load_state_dict(state_dict)

agent = agent.to(device)

env = FragmentBuilderEnv(protected_fragments=final_fragments,
                         z_table=z_table,
                        max_episode_steps=n_steps,
                        valid_action_masks=valid_action_masks,
                        embed_hydrogens=EMBED_HYDROGENS)

mws = [round(MolWt(fragment)) for fragment in protected_fragments]
mws = np.array(mws)

rows = []
for seed_i, seed in enumerate(tqdm(all_seeds)):

    logging.info(seed_i)
    logging.info(complx.vina_protein.pdbqt_filepath)
    
    complx_i = seed_to_complex[seed_i]
    complx = complexes[complx_i]
    ap_label = list(seed.construct.get_attach_points().values())[0]
    valid_actions = torch.nonzero(valid_action_masks[ap_label])
    valid_actions = valid_actions.reshape(-1)
    
    mols = []
    for action_i in valid_actions:
        env.reset(complx=complx,
                  seed=seed.construct,
                  scorer=None,
                  seed_i=0)
        
        env.step(int(action_i))
        mol = env.get_clean_mol(env.seed)
        mols.append(mol)

    vina_cli = VinaCLI()
    receptor_paths = [complx.vina_protein.pdbqt_filepath] * len(mols)
    native_ligands = [complx.ligand] * len(mols)
    scores = vina_cli.get(receptor_paths=receptor_paths,
                          native_ligands=native_ligands,
                          ligands=mols)
    scores = np.array(scores)

    if np.all(scores == 0):
        continue
    
    logging.info(scores)

    with torch.no_grad():
        obs = env._get_obs()
        current_obs = [obs]
        x = Batch.from_data_list(data_list=current_obs)
        x.to(device)
        features = agent.extract_features(x)
        frag_logits = agent.actor(features).reshape(-1)
        frag_logits = torch.tanh(frag_logits)
        frag_logits = frag_logits * 3
        valid_logits = frag_logits[valid_action_masks[ap_label]]
        valid_logits = valid_logits.cpu()

    logging.info(valid_logits)
    
    native_i = protected_fragments_smiles.index(Chem.MolToSmiles(seed.removed_fragment))
    valid_native_i = torch.nonzero(valid_actions == native_i).item()
    native_score = scores[valid_native_i]
    native_score_rank = np.nonzero(scores.argsort() == valid_native_i)[0].item()
    native_score_nrank = native_score_rank / len(scores)

    probs = torch.nn.functional.softmax(valid_logits)
    native_logit = valid_logits[valid_native_i]
    native_prob = probs[valid_native_i]
    native_logit_rank = np.nonzero((-valid_logits).argsort() == valid_native_i)[0].item()
    native_logit_nrank = native_logit_rank / len(valid_logits)

    selected = probs > 0.001
    scores_selected = scores[selected]
    scores_nonselected = scores[~selected]
    diff_median = np.median(scores_selected) - np.median(scores_nonselected)
    
    better = scores < native_score
    better_selected = scores_selected < native_score
    better_nonselected = scores_nonselected < native_score

    spear_coeff = spearmanr(scores, -valid_logits)
    logging.info(spear_coeff)
    
    row = {'seed_i': seed_i,
          'filepath': complx.vina_protein.pdbqt_filepath,
          'removed_fragment': Chem.MolToSmiles(seed.removed_fragment),
          'native_score': native_score,
          'native_score_nrank': native_score_nrank,
          'native_logit': native_logit.item(),
          'native_prob': native_prob.item(),
          'native_logit_nrank': native_logit_nrank,
          'diff_median': diff_median,
          'spear_coeff': spear_coeff.statistic,
          'better_all': better.sum(),
          'better_selected': better_selected.sum(),
          'better_nonselected': better_nonselected.sum(),
          'n_selected': selected.sum().item(),
          'n_frags': len(scores)}
    rows.append(row)
    
    logging.info(row)
    
df = pd.DataFrame(rows)
df.to_csv('performances.csv')