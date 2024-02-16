import torch
import torch.nn.functional as F
import os

from rdkit import Chem # safe import before any ccdc import
from rdkit.Chem import Mol

from torch_geometric.data import Batch, Data
# Differentiate PyGDataLoader and torch DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Any, Dict, List
from .atomistic import AtomisticNN
from abc import abstractclassmethod
from ymir.params import FEATURES_DIM
from torch_geometric.utils import unbatch
from torch_scatter import scatter_mean, scatter_add


class AtomisticNNModel():
    """
    Uses an atomistic neural network (NN) as a model to process an input conformation 
    to obtain a predicted value. In current work, we try to predict the ARMSD to 
    bioactive conformation
    
    :param config: Parameters of the model. Current list is:
        "num_interactions": int = 6, number of interaction blocks
        "cutoff": float = 10, cutoff for neighbourhood convolution graphs
        "lr": float = 1e-5, learning rate
        'batch_size': int = 256, batch size
        'data_split': DataSplit, object to split the dataset stored in the model
    :type config: Dict[str, Any]
    """
    
    def __init__(self, 
                 atomistic_nn: AtomisticNN,
                 config: Dict[str, Any] = None,
                 device: torch.device = torch.device('cuda'),
                #  action_dim: int
                 ):
        self.config = config
        self.atomistic_nn = atomistic_nn
        self.device = device
        self.atomistic_nn.to(device)
    
    @property
    def name(self) -> str:
        """
        :return: Name of the model
        :rtype: str
        """
        return 'AtomisticNNModel'


    def forward(self, 
                batch: Batch) -> torch.Tensor:
        """
        Forward pass
        
        :param batch: PyG batch of atoms
        :type batch: torch_geometric.data.Batch
        :return: Predicted values
        :rtype: torch.Tensor (n_confs)
        """
        atomic_contributions = self.get_atomic_contributions(batch)
        
        # # get last vector per instance
        # unbatched_contributions = unbatch(atomic_contributions, batch=batch.batch)
        
        # env_list = []
        # for contrib in unbatched_contributions:
        #     env_list.append(contrib[-1]) # we assume the new frag environement is the last one per batch
        #     # might need to be modified to fit the Aggregation paradigm in torch geometric
        
        # env_vector = torch.stack(env_list)
        
        env_vector = scatter_mean(src=atomic_contributions, 
                                index=batch.batch, 
                                dim=0,)
        
        # import pdb;pdb.set_trace()
        
        return env_vector
        
        # pred = self.lin_action(atomic_contributions)
        # mask_value = torch.tensor(
        #     torch.finfo(pred.dtype).min, dtype=pred.dtype
        # )
        # logits = torch.where(batch.mask, logits, mask_value)
        # probs = torch.softmax(logits)
        # return probs


    def record_loss(self,
                    batch: Batch,
                    loss_name: str = 'train_loss'
                    ) -> torch.Tensor:
        """
        Perform forward pass and records then returns the loss.
        Loss is MSELoss
        
        :param batch: PyG batch of atoms
        :type batch: torch_geometric.data.Batch
        :param loss_name: Name of the computed loss. Depends on whether the model
        is training, validating or testing
        :type loss_name: str
        :return: Loss
        :rtype: torch.tensor (1)
        """
        pred = self.forward(batch)
        # import pdb;pdb.set_trace()
        y = batch.rmsd
        # y = batch.armsd
        
        # ARMSD can be -1 in case of error in splitting
        if not torch.all(y >= 0):
            import pdb;pdb.set_trace()
        loss = F.mse_loss(pred.squeeze(), y)
        self.log(loss_name, loss.squeeze(), batch_size=self.batch_size)
        return loss


    def get_preds_for_data_list(self,
                                data_list: List[Data]) -> None:
        """
        Get predictions for data_list (output from featurizer)
        
        :param data_list: List of PyG Data
        :type data_list: List[Data]
        :return: Predictions for each conformation
        :rtype: torch.Tensor (n_confs)
        """
        with torch.inference_mode():
            data_loader = PyGDataLoader(data_list, 
                                        batch_size=self.batch_size)
            preds = None
            for batch in data_loader:
                batch.to(self.device)
                pred = self(batch)
                pred = pred.detach() # .cpu().numpy().squeeze(1)
                if preds is None:
                    preds = pred
                else:
                    preds = torch.cat((preds, pred))
            return preds


    def get_atomic_contributions(self, 
                                 batch) -> torch.Tensor:
        """
        Performs atomistic NN forward to obtain atomic contributions
        
        :param batch: Batch of featurized conformations
        :type batch: torch_geometric.data.Batch
        :return: Atomic contributions
        :rtype: torch.Tensor (n_atoms)
        """
        return self.atomistic_nn.forward(z=batch.x.squeeze().long(), 
                                            pos=batch.pos,
                                            batch=batch.batch)
        

    def __call__(self, 
                 batch,
                 *args: Any, 
                 **kwds: Any) -> torch.Tensor:
        return self.forward(batch,
                            *args, 
                            **kwds)