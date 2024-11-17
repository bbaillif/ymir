import torch
import random
import numpy as np
import logging
import pickle
import os
import pandas as pd
# import wandb

from tqdm import tqdm
from rdkit import Chem
from datetime import datetime
from ymir.fragment_library import FragmentLibrary
from ymir.data.structure import Complex
from ymir.utils.fragment import (get_masks,
                                 ConstructionSeed)

from ymir.atomic_num_table import AtomicNumberTable

from ymir.params import (EMBED_HYDROGENS, 
                         HIDDEN_IRREPS,
                         IRREPS_OUTPUT,
                         SEED,
                         POCKET_RADIUS)
from torch.utils.data import random_split, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from regressor import SOAPFeaturizer, SOAPScoreDataset, LitModel, GraphDataset, GraphFeaturizer, LitCNNModel, LitComENet
from sklearn.feature_selection import VarianceThreshold

# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader 

logging.basicConfig(filename='train_regressor.log', 
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
torch.set_float32_matmul_precision('high')
generator = torch.Generator().manual_seed(42)

# 1 episode = grow fragments + update NN
n_epochs = 10_000
batch_size = 128 # NN batch, input is Data, output are actions + predicted reward
# lr = 1e-3
lr = 1e-4
device = torch.device('cuda')

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = f"ymir_regressor_{timestamp}"

# writer = SummaryWriter(f"logs_regression/{experiment_name}")

removeHs = not EMBED_HYDROGENS
fragment_library = FragmentLibrary(removeHs=removeHs)

z_list = [0, 6, 7, 8, 16, 17]
if EMBED_HYDROGENS:
    z_list.append(1)
z_table = AtomicNumberTable(zs=z_list)

# Remove fragment having at least one heavy atom not in list
protected_fragments = fragment_library.get_restricted_fragments(z_list)
           
protected_fragments_smiles = [Chem.MolToSmiles(frag) for frag in protected_fragments]

valid_action_masks = get_masks(protected_fragments)

# soap_featurizer = SOAPFeaturizer()
graph_featurizer = GraphFeaturizer(z_table)

pocket_radius = POCKET_RADIUS

lp_pdbbind = pd.read_csv('LP_PDBBind.csv')
lp_pdbbind = lp_pdbbind.rename({'Unnamed: 0' : 'PDB_ID'}, axis=1)

# n_samples = 2500
# input_soaps_path = f'/home/bb596/hdd/ymir/input_soaps_{n_samples}.p'
input_data_list_path = f'/home/bb596/hdd/ymir/data/input_data_{pocket_radius}.p'
# output_scores_path = f'/home/bb596/hdd/ymir/data/output_scores.p'
# output_masks_path = f'/home/bb596/hdd/ymir/data/output_masks.p'
# subset_path = f'/home/bb596/hdd/ymir/data/subsets.p'

pre_computed_data = os.path.exists(input_data_list_path)

if pre_computed_data:
    with open(input_data_list_path, 'rb') as f:
        input_data_list = pickle.load(f)
else:
    input_data_list = []

# input_soaps = []
output_scores = []
output_masks = []
subsets = []

dataset_path = '/home/bb596/hdd/ymir/dataset/'
data_filenames = os.listdir(dataset_path)
for data_filename in tqdm(data_filenames):
    
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
            
            # soap = soap_featurizer.featurize_complex(construct, 
            #                                         pocket_mol,
            #                                         center_pos)
            # input_soaps.append(soap)

            if not pre_computed_data:
                pocket_mol = complx.pocket.mol
                data = graph_featurizer.featurize_complex(construct, 
                                                        pocket_mol,
                                                        center_pos)
                input_data_list.append(data)

            relative_scores = [score - native_score for score in absolute_scores]
            valid_action_mask = valid_action_masks[attach_label]
            output_masks.append(valid_action_mask)
            
            final_scores = []
            n = 0
            for b in valid_action_mask:
                if b:
                    final_scores.append(relative_scores[n])
                    n += 1
                else:
                    final_scores.append(0)
            output_scores.append(final_scores)
            
            subsets.append(subset)
    
# with open(input_soaps_path, 'wb') as f:
#     pickle.dump(input_soaps, f)
if not pre_computed_data:
    with open(input_data_list_path, 'wb') as f:
        pickle.dump(input_data_list, f)
# with open(output_scores_path, 'wb') as f:
#     pickle.dump(output_scores, f)
# with open(output_masks_path, 'wb') as f:
#     pickle.dump(output_masks, f)
        
# else:
    # with open(input_soaps_path, 'rb') as f:
    #     input_soaps = pickle.load(f)
    
    # with open(output_scores_path, 'rb') as f:
    #     output_scores = pickle.load(f) 
    # with open(output_masks_path, 'rb') as f:
    #     output_masks = pickle.load(f)

# input_soaps = np.array(input_soaps)
# vt = VarianceThreshold()
# selected_soaps = vt.fit_transform(input_soaps)

# input_soaps = torch.tensor(selected_soaps, dtype=torch.float)

output_scores = torch.tensor(np.array(output_scores), dtype=torch.float)
output_masks = torch.tensor(np.array(output_masks), dtype=torch.bool)

# input_size = input_soaps.shape[-1]
output_size = output_scores.shape[-1]

# model = LitModel(input_size, output_size, lr)
model = LitCNNModel(output_size=output_size,
                 hidden_irreps=HIDDEN_IRREPS,
                 irreps_output=IRREPS_OUTPUT,
                 z_table=z_table,
                 lr=lr)
# model = LitComENet(output_size=output_size,
#                    lr=lr)

# dataset = SOAPScoreDataset(input_soaps, output_scores, output_masks)
dataset = GraphDataset(input_data_list, output_scores, output_masks)
    
train_idxs = [i for i, subset in enumerate(subsets) if subset == 'train']
val_idxs = [i for i, subset in enumerate(subsets) if subset == 'val']
    
# train_idx, val_idx, test_idx = random_split(range(len(dataset)), [0.6, 0.2, 0.2], generator=generator)
training_set = Subset(dataset, train_idxs)
validation_set = Subset(dataset, val_idxs)
# test_set = Subset(dataset, test_idx)

# batch_size = 256
batch_size = 48

train_loader = DataLoader(training_set, batch_size=batch_size, num_workers=4)
val_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=4)

early_stopping = EarlyStopping('val_loss', patience=10)
checkpoint_filename = experiment_name + '-{epoch}-{val_loss:.2f}'
model_checkpoint = ModelCheckpoint(dirpath=f'/home/bb596/hdd/ymir/models/',
                                   filename=checkpoint_filename,
                                   monitor='val_loss')
logger = TensorBoardLogger("logs_regression", name=experiment_name)

callbacks = [
    early_stopping, 
    model_checkpoint
    ]
trainer = Trainer(max_epochs=n_epochs,
                  accelerator='gpu',
                  callbacks=callbacks,
                  logger=logger,
                  precision=32,
                  log_every_n_steps=1)
trainer.fit(model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader)

import pdb;pdb.set_trace()