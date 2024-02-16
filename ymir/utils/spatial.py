import numpy as np

from rdkit.Chem import Conformer, rdMolTransforms
from scipy.spatial.transform import Rotation
from typing import Sequence

def rotate_conformer(conformer: Conformer,
                        rotation: Rotation) :
    rotation_matrix = rotation.as_matrix()
    transf_matrix = [[*rotation_matrix[0],0],
                    [*rotation_matrix[1],0],
                    [*rotation_matrix[2],0],
                    [0,0,0,1]]
    transf_matrix = np.array(transf_matrix)
    rdMolTransforms.TransformConformer(conformer, transf_matrix)
    
    
def translate_conformer(conformer: Conformer,
                        translation: Sequence[float]):
    
    transf_matrix = [[1,0,0,translation[0]],
                     [0,1,0,translation[1]],
                     [0,0,1,translation[2]],
                     [0,0,0,1]]
    transf_matrix = np.array(transf_matrix)
    rdMolTransforms.TransformConformer(conformer, transf_matrix)