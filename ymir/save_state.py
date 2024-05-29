from typing import NamedTuple
from rdkit.Chem import Mol
from ymir.data import Fragment
from ymir.utils.spatial import Transformation

class StateSave(NamedTuple):
    score: float
    rmsd: float
    seed: Fragment
    
Memory = dict[tuple, StateSave]