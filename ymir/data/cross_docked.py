import logging
import os
import pickle

from tqdm import tqdm
from ymir.params import (CROSSDOCKED_SPLITS_P_PATH, 
                               CROSSDOCKED_POCKET10_PATH, 
                               BENCHMARK_DIRPATH, 
                               CROSSDOCKED_DATA_PATH,
                               MINIMIZED_DIRPATH)
from rdkit import Chem
from rdkit.Chem import Mol
from ymir.complex_minimizer import ComplexMinimizer

CROSSDOCKED_SUBSETS = ['train', 'test']
CROSSDOCKED_LIGANDS_PATH = os.path.join(BENCHMARK_DIRPATH, 'crossdocked_ligands.sdf')

# ligand_filename is actually TARGET_NAME/ligand_filename.sdf

class CrossDocked():
    
    def __init__(self,
                 name: str = 'CrossDocked',
                 root: str = BENCHMARK_DIRPATH,
                 pocket_path: str = CROSSDOCKED_POCKET10_PATH,
                 data_path: str = CROSSDOCKED_DATA_PATH,
                 split_path: str = CROSSDOCKED_SPLITS_P_PATH,
                 minimized_path: str = MINIMIZED_DIRPATH,
                 subset: str = 'train',
                 ) -> None:
        assert subset in CROSSDOCKED_SUBSETS, \
            f'subset must be in {CROSSDOCKED_SUBSETS}'
        self.name = name
        self.root = root
        self.pocket_path = pocket_path
        self.data_path = data_path
        self.split_path = split_path
        self.minimized_path = minimized_path
        self.subset = subset
        
        self.ligands_path = os.path.join(self.root, f'crossdocked_ligands_{self.subset}.sdf')
    
    
    def get_ligands(self):
        if not os.path.exists(self.ligands_path):
            ligands = self.compile_ligands()
            ligands = iter(ligands)
        else:
            ligands = [ligand for ligand in Chem.SDMolSupplier(self.ligands_path)]
        return ligands
    
    
    def get_split(self) -> list[tuple[str, str]]:
        with open(self.split_path, 'rb') as f:
            splits = pickle.load(f)
        return splits[self.subset]
    
    
    def compile_ligands(self):
        logging.info('Compiling CrossDocked ligands')
        ligands = []
        split_data = self.get_split()
        writer = Chem.SDWriter(self.ligands_path)
        for duo in tqdm(split_data):
            pocket_path, ligand_subpath = duo
            ligand_path = os.path.join(self.pocket_path, ligand_subpath)
            ligand = next(Chem.SDMolSupplier(ligand_path))
            # if ligand is not None:
            #     ligands.append(ligand)
            #     writer.write(ligand)
            # else:
            #     logging.warning('Invalid CrossDocked ligand')
            ligands.append(ligand)
            writer.write(ligand)
        return ligands
                
    
    def __iter__(self) -> list[Mol]:
        return self.get_ligands()
    
    
    def get_native_ligand(self,
                          ligand_filename: str):
        native_ligand_filepath = os.path.join(self.data_path, ligand_filename)
        native_ligand = [mol for mol in Chem.SDMolSupplier(native_ligand_filepath, removeHs=False)][0]
        return native_ligand
    
    def get_ligand_filenames(self):
        split = self.get_split()
        return [t[1] for t in split]
    
    def get_minimized_native(self,
                             ligand_filename: str,
                             complex_minimizer: ComplexMinimizer):
        target_dirname, real_ligand_filename = ligand_filename.split('/') 
        minimized_target_path = os.path.join(self.minimized_path, target_dirname)
        if not os.path.exists(minimized_target_path):
            os.mkdir(minimized_target_path)
        
        minimized_filename = 'generated_' + real_ligand_filename.replace('.sdf', 
                                                                        f'_native_minimized.sdf')
        minimized_filepath = os.path.join(minimized_target_path,
                                            minimized_filename)
        
        native_ligand = self.get_native_ligand(ligand_filename)
        
        # Minimize native ligand
        if not os.path.exists(minimized_filepath):
            logging.info(f'Minimized native ligand in {minimized_filepath}')
            mini_native_ligand = complex_minimizer.minimize_ligand(native_ligand)
            with Chem.SDWriter(minimized_filepath) as writer:
                writer.write(mini_native_ligand)
        else:
            logging.info(f'Loading minimized native ligand from {minimized_filepath}')
            mini_native_ligand = Chem.SDMolSupplier(minimized_filepath, removeHs=False)[0]
            
        return mini_native_ligand
    
    
    def get_filename_i(self,
                       ligand_filename: str):
        ligand_filenames = self.get_ligand_filenames()
        return ligand_filenames.index(ligand_filename)
    
    
    def get_original_structure_path(self,
                                    ligand_filename: str):
        target_dirname, real_ligand_filename = ligand_filename.split('/') 
        pdb_filename = f'{real_ligand_filename[:10]}.pdb'
        original_structure_path = os.path.join(self.data_path, 
                                               target_dirname, 
                                               pdb_filename)
        return original_structure_path