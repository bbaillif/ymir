from abc import abstractmethod
from typing import Sequence, Any

class MolFeaturizer():
    
    @abstractmethod
    def featurize_mol(self,
                      mol) -> Sequence[Any]:
        pass