import torch

from torch import nn, optim
from ymir.params import EMBED_HYDROGENS
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
from ase.io import read
from torch.utils.data import random_split, DataLoader, Dataset
from pytorch_lightning import LightningModule
from ymir.data import Fragment
from ymir.featurizer_sn import Featurizer
from ymir.model import CNN, ComENetModel
from ymir.atomic_num_table import AtomicNumberTable
from rdkit import Chem
from rdkit.Chem import Mol
from torch_geometric.data import Data, InMemoryDataset
from e3nn import o3

class LinearLayers(nn.Module):
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.sequential = nn.Sequential(nn.Linear(input_size, input_size),
                                        nn.SiLU(),
                                        nn.Linear(input_size, input_size // 2),
                                        nn.SiLU(),
                                        nn.Linear(input_size // 2, output_size))
        
    def forward(self,
                x):
        return self.sequential(x)
    
    
class LitModel(LightningModule):
    
    def __init__(self, input_size: int,
                 output_size: int,
                 lr: float):
        super().__init__()
        self.model = LinearLayers(input_size, output_size)
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction='none')
        # self.mse_loss = nn.MSELoss()


    def get_loss(self, target, predict, mask):
        loss = self.mse_loss(target, predict)
        loss = (loss * mask.float()).sum() # gives \sigma_euclidean over unmasked elements
        non_zero_elements = mask.sum()
        loss = loss / non_zero_elements
        return loss

    def training_step(self, batch, batch_idx):
        soaps, scores, mask = batch
        predict = self.model(soaps)
        
        loss = self.get_loss(scores, predict, mask)
        
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        soaps, scores, mask = batch
        predict = self.model(soaps)
        
        loss = self.get_loss(scores, predict, mask)
        
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    
class CNNModel(nn.Module):
    
    def __init__(self,
                 output_size: int,
                 hidden_irreps: o3.Irreps,
                 irreps_output: o3.Irreps,
                 z_table: AtomicNumberTable,
                 ) -> None:
        super().__init__()
        self.output_size = output_size
        self.hidden_irreps = hidden_irreps
        self.irreps_output = irreps_output
        self.z_table = z_table
        self.cnn = CNN(hidden_irreps=self.hidden_irreps,
                        irreps_output=self.irreps_output,
                        num_elements=len(self.z_table))
        self.linear_layers = LinearLayers(self.irreps_output.dim, output_size)
    
    def forward(self, x):
        features = self.cnn(x)
        return self.linear_layers(features)
    
    
class LitCNNModel(LightningModule):
    
    def __init__(self, 
                 output_size: int,
                 hidden_irreps: o3.Irreps,
                 irreps_output: o3.Irreps,
                 z_table: AtomicNumberTable,
                 lr: float):
        super().__init__()
        self.model = CNNModel(output_size, hidden_irreps,
                              irreps_output, z_table)
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction='none')
        # self.mse_loss = nn.MSELoss()


    def get_loss(self, target, predict, mask):
        loss = self.mse_loss(target, predict)
        loss = (loss * mask.float()).sum() # gives \sigma_euclidean over unmasked elements
        non_zero_elements = mask.sum()
        loss = loss / non_zero_elements
        return loss

    def training_step(self, batch, batch_idx):
        data, scores, mask = batch
        predict = self.model(data)
        
        loss = self.get_loss(scores, predict, mask)
        
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        data, scores, mask = batch
        predict = self.model(data)
        
        loss = self.get_loss(scores, predict, mask)
        
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    
class LitComENet(LightningModule):
    
    def __init__(self, 
                 output_size: int,
                 lr: float):
        super().__init__()
        self.model = ComENetModel()
        self.linear_layers = LinearLayers(self.model.features_dim, output_size)
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction='none')
        # self.mse_loss = nn.MSELoss()


    def get_loss(self, target, predict, mask):
        loss = self.mse_loss(target, predict)
        loss = (loss * mask.float()).sum() # gives \sigma_euclidean over unmasked elements
        non_zero_elements = mask.sum()
        loss = loss / non_zero_elements
        return loss

    def training_step(self, batch, batch_idx):
        data, scores, mask = batch
        features = self.model(data)
        predict = self.linear_layers(features)
        
        loss = self.get_loss(scores, predict, mask)
        
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        data, scores, mask = batch
        features = self.model(data)
        predict = self.linear_layers(features)
        
        loss = self.get_loss(scores, predict, mask)
        
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    
class SOAPFeaturizer():
    
    def __init__(self,
                 embed_hydrogens: bool = EMBED_HYDROGENS) -> None:
        
        self.embed_hydrogens = embed_hydrogens
        
        species = ['C', 'N', 'O', 'Cl', 'S']
        if self.embed_hydrogens:
            species = ['H'] + species
        self.soap = SOAP(species=species,
                        periodic=False,
                        r_cut=8,
                        n_max=4,
                        l_max=4)
        
        
    def featurize_complex(self,
                          ligand_frag: Fragment,
                          pocket_mol: Mol,
                          center_pos: list[float]):
        seed_copy = Fragment(ligand_frag, ligand_frag.protections)
        seed_copy.protect()
        seed_atoms = self.rdkit_to_ase(seed_copy)
        pocket_atoms = self.rdkit_to_ase(pocket_mol)
        if not self.embed_hydrogens:
            seed_atoms = seed_atoms[[atom.index for atom in seed_atoms if atom.symbol != 'H']]
            pocket_atoms = pocket_atoms[[atom.index for atom in pocket_atoms if atom.symbol != 'H']]
        
        total_atoms = seed_atoms + pocket_atoms
        seed_soap = self.soap.create(total_atoms, centers=[center_pos])

        # seed_soap = normalize(seed_soap)

        return seed_soap.squeeze()
    
    def rdkit_to_ase(self,
                     mol: Mol):
        filename = 'mol.xyz'
        Chem.MolToXYZFile(mol, filename)
        ase_atoms = read(filename)
        return ase_atoms    


class GraphFeaturizer():
    
    def __init__(self,
                 z_table,
                 embed_hydrogens: bool = EMBED_HYDROGENS) -> None:
        self.featurizer = Featurizer(z_table)
        self.embed_hydrogens = embed_hydrogens


    def featurize_complex(self,
                          ligand_frag: Fragment,
                          pocket_mol: Mol,
                          center_pos: list[float]):
        
        ligand_x, ligand_pos = self.featurizer.get_fragment_features(fragment=ligand_frag, 
                                                                    embed_hydrogens=self.embed_hydrogens,
                                                                    center_pos=center_pos)
        protein_x, protein_pos = self.featurizer.get_mol_features(mol=pocket_mol,
                                                                embed_hydrogens=self.embed_hydrogens,
                                                                center_pos=center_pos)
        
        pocket_x = protein_x + ligand_x
        pocket_pos = protein_pos + ligand_pos
        
        mol_id = [0] * len(protein_x) + [1] * len(ligand_x)
        
        # x = torch.tensor(pocket_x, dtype=torch.float)
        x = torch.tensor(pocket_x, dtype=torch.long)
        pos = torch.tensor(pocket_pos, dtype=torch.float)
        mol_id = torch.tensor(mol_id, dtype=torch.long)
        data = Data(x=x,
                    pos=pos,
                    mol_id=mol_id
                    )
        return data
    
    

    
class SOAPScoreDataset(Dataset):

    def __init__(self, soaps, scores, masks) -> None:
        super().__init__()
        self.soaps = soaps
        self.scores = scores
        self.masks = masks
        
    def __len__(self):
        return self.soaps.shape[0]
    
    def __getitem__(self, index):
        return self.soaps[index], self.scores[index], self.masks[index]
    
    
class GraphDataset(Dataset):
    def __init__(self, data_list, scores, masks) -> None:
        super().__init__()
        self.data_list = data_list
        self.scores = scores
        self.masks = masks
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index], self.scores[index], self.masks[index]