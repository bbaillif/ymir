import torch

from typing import Dict, Any
from .atomistic.comenet import AtomicComENet
from .atomistic_nn_model import AtomisticNNModel
from ymir.params import COMENET_MODEL_NAME, FEATURES_DIM
from torch import nn

class ComENetModel(AtomisticNNModel):
    """
    Class to setup an AtomisticNNModel using SchNet as backend
    
    :param config: Dictionnary of parameters. Must contain:
        num_interations: int = number of interation blocks
        cutoff: int = distance cutoff in Angstrom for neighbourhood convolutions
    :type config: Dict[str, Any]
    :param readout: Type of aggregation for atoms in a molecule. Default: sum
    :type readout:str
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 readout: str = 'add',
                 features_dim: int = FEATURES_DIM,
                 device: torch.device = torch.device('cuda')):
        self.features_dim = features_dim
        atomistic_nn = AtomicComENet(readout=readout,
                                      features_dim=features_dim)
        AtomisticNNModel.__init__(self,
                                  atomistic_nn=atomistic_nn,
                                  config=config,
                                  device=device)
        
    @property
    def name(self) -> str:
        """
        :return: Name of the model
        :rtype: str
        """
        return COMENET_MODEL_NAME
    