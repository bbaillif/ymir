import numpy as np

from rdkit import Chem
from rdkit.Chem import Conformer, rdMolTransforms
from scipy.spatial.transform import Rotation
from typing import Union

Translation = list[float] # shape: (3,)
Transformation = Union[Rotation, Translation]

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
                        translation: Translation):
    
    transf_matrix = [[1,0,0,translation[0]],
                     [0,1,0,translation[1]],
                     [0,0,1,translation[2]],
                     [0,0,0,1]]
    transf_matrix = np.array(transf_matrix)
    rdMolTransforms.TransformConformer(conformer, transf_matrix)
    
    
def apply_transformation(conformer: Conformer,
                         transformation: Transformation):
    if isinstance(transformation, Rotation):
        rotation_inv = transformation.inv()
        rotate_conformer(conformer, rotation_inv)
    else:
        translation_inv = -transformation
        translate_conformer(conformer, translation_inv)
    
    
def reverse_transformations(conformer: Conformer,
                            transformations: list[Transformation]):
    for transformation in reversed(transformations):
        apply_transformation(conformer, transformation)
        
        
def rdkit_distance_matrix(mol1: Conformer, mol2: Conformer) -> np.ndarray:
    combined_mol = Chem.CombineMols(mol1, mol2)
    distance_matrix = Chem.Get3DDistanceMatrix(combined_mol)
    return distance_matrix


def add_noise(conformer: Conformer,
              noise_std: float = 0.01):
    num_atoms = conformer.GetNumAtoms()
    for i in range(num_atoms):
        initial_pos = conformer.GetAtomPosition(i)
        noisy_pos = initial_pos + noise_std * np.random.randn(3)
        conformer.SetAtomPosition(i, noisy_pos)