from typing import Sequence

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
                   atomic_number: int) -> int:
        return self.zs.index(atomic_number)