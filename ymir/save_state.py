from typing import NamedTuple
from rdkit.Chem import Mol
from ymir.data import Fragment
from ymir.utils.spatial import Transformation

class StateSave(NamedTuple):
    score: float
    seed: Fragment
    pocket_mol: Mol
    transformations: list[Transformation]
    
Memory = dict[tuple, StateSave]