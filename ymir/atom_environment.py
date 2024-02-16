import numpy as np
import torch

from rdkit.Chem import Mol
from e3nn import o3, io
from e3nn.io import SphericalTensor
from typing import Sequence
from torch_cluster import radius
from collections import defaultdict
from ymir.params import LMAX

class AtomicNumberTable():
    # Copied from mace/tools/utils
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, 
                   index: int) -> int:
        return self.zs[index]

    def z_to_index(self, 
                   atomic_number: str) -> int:
        return self.zs.index(atomic_number)


class SHEnvironment():
    
    def __init__(self,
                 z_table: AtomicNumberTable,
                 lmax: int = LMAX,
                 rmax: float = 7.0,
                 device: torch.device = 'cuda'
                 ) -> None:
        self.z_table = z_table
        self.lmax = lmax
        # self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        # self.spherical_harmonics = o3.SphericalHarmonics(
        #     self.sh_irreps, normalize=True, normalization="component"
        # )
        self.spherical_tensor = SphericalTensor(lmax, +1, -1)
        self.rmax = rmax
        self.device = device
        
    def get_environment(self,
                        mol: Mol,
                        atom_id: int):
        
        assert atom_id < mol.GetNumAtoms(), 'Atom ID not in molecule'
        
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atomic_numbers = torch.tensor(atomic_numbers)
        positions = mol.GetConformer().GetPositions()
        positions = torch.tensor(positions)
        
        other_atomic_numbers = torch.cat([atomic_numbers[:atom_id], atomic_numbers[atom_id+1:]])
        center_position = positions[atom_id].reshape((1, 3))
        other_positions = torch.cat([positions[:atom_id], positions[atom_id+1:]])
        
        # edge is i->j, source->target
        # edge_index is (j, i)
        edge_index = radius(center_position, 
                            other_positions,
                            self.rmax)
        tgt, src = edge_index
        assert (src == 0).all()
        
        graph_atomic_numbers = other_atomic_numbers[tgt]
        graph_positions = other_positions[tgt]
        
        vectors = center_position - graph_positions
        lengths = torch.linalg.norm(vectors, dim=-1)
        values = (self.rmax - lengths) / self.rmax
        
        n_atomic_numbers = len(self.z_table)
        n_sh_elements = self.spherical_tensor.dim
        environment = torch.zeros((n_atomic_numbers, n_sh_elements))
        
        for atomic_number in set(graph_atomic_numbers):
            z_idx = self.z_table.z_to_index(atomic_number)
            graph_indices = (graph_atomic_numbers == atomic_number)
            z_vectors = vectors[graph_indices]
            z_values = values[graph_indices]
            # import pdb;pdb.set_trace()
            environment[z_idx] = self.spherical_tensor.from_samples_on_s2(z_vectors, z_values)
            
        return environment