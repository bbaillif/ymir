import torch
from abc import ABC, abstractmethod
from ymir.params import FEATURES_DIM

class AtomisticNN(ABC):
    """
    Base class for atomistic neural network: taking as input a conformation
    and returning values for each atom of the molecule
    
    :param readout: Readout function to perform on the list of individual
        atomic values
    :type readout: str
    """
    
    def __init__(self,
                 readout: str = 'sum',
                 features_dim: int = FEATURES_DIM) -> None:
        self.readout = readout
        self.features_dim = features_dim
        
    
    @abstractmethod
    def forward(self,
                z: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        pass