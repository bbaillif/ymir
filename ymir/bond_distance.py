import os
import numpy as np
import pickle

from ymir.geometry import CSDGeometry
from ymir.params import MBD_PATH
from collections import defaultdict

class MedianBondDistance():
    
    def __init__(self,
                 mbd_path: str = MBD_PATH,
                 ) -> None:
        self.mbd_path = mbd_path
        
        if not os.path.exists(mbd_path):
            self.mbd = self.compute_bd()
            with open(mbd_path, 'wb') as f:
                pickle.dump(self.mbd, f)
        else:
            with open(mbd_path, 'rb') as f:
                self.mbd = pickle.load(f)
                
                
    def get_mbd(self,
                atom_symbol0: str,
                atom_symbol1: str) -> float:
        
        if atom_symbol0 < atom_symbol1:
            atom_symbols = (atom_symbol0, atom_symbol1)
        else:
            atom_symbols = (atom_symbol1, atom_symbol0)
            
        mbd = None
        if atom_symbols in self.mbd:
            mbd = self.mbd[atom_symbols]
            
        return mbd
        
    def compute_bd(self) -> dict[tuple[str, str], float]:
        
        bds: dict[tuple[str, str], list[float]] = defaultdict(list)
        csd_geometry = CSDGeometry()
        values = csd_geometry.read_values()
        bond_values = values['bond']
        for pattern, values in bond_values.items():
            neighbors0, atom0, sep, atom1, neighbors1 = pattern
            if sep == '-': # only get single bond
                atom_symbol0: str = atom0[0]
                atom_symbol1: str = atom1[0]
                if atom_symbol0 < atom_symbol1:
                    atom_symbols = (atom_symbol0, atom_symbol1)
                else:
                    atom_symbols = (atom_symbol1, atom_symbol0)
                bds[atom_symbols].extend(values)
            
        median_bds = {}
        for atom_symbols, values in bds.items():
            # if 'C' in atom_symbols:
            #     import pdb;pdb.set_trace()
            median_bds[atom_symbols] = np.median(values)
            
        return median_bds