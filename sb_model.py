import pickle
import os
import logging
import gzip

from rdkit import Chem
from rdkit.Chem import Mol
from abc import ABC, abstractmethod
from genbench3d.data import ComplexMinimizer
from genbench3d.sb_model import SBModel

    
class Ymir(SBModel):
    
    def __init__(self,
                 minimized_path: str,
                 gen_path: str = '/home/bb596/hdd/ymir/generated_mols_cd0/',
                 name: str = 'Ymir',
                 ) -> None:
        super().__init__(name=name,
                         minimized_path=minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        target_path, real_ligand_filename = ligand_filename.split('/')
        gen_mols_filename = 'generated_' + real_ligand_filename + '.sdf'
        gen_mols_filepath = os.path.join(self.gen_path, target_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in Ymir results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
            
        has_attach_l = []
        for gen_mol in gen_mols:
            has_attach = False
            for atom in gen_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    has_attach = True
            has_attach_l.append(has_attach)
            Chem.SanitizeMol(gen_mol)
            Chem.AssignStereochemistry(gen_mol)
                    
        logging.info(f'{ligand_filename}: has attach: {sum(has_attach_l)}')
    
        return gen_mols
    
    
class YmirRandom(SBModel):
    
    def __init__(self,
                 minimized_path: str,
                 name: str = 'YmirRandom',
                 gen_path = '/home/bb596/hdd/ymir/generated_mols_random/'
                 ) -> None:
        super().__init__(name=name,
                         minimized_path=minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        target_path, real_ligand_filename = ligand_filename.split('/')
        gen_mols_filename = 'generated_' + real_ligand_filename + '.sdf'
        gen_mols_filepath = os.path.join(self.gen_path, target_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in Ymir results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
            
        has_attach_l = []
        for gen_mol in gen_mols:
            has_attach = False
            for atom in gen_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    has_attach = True
            has_attach_l.append(has_attach)
            Chem.SanitizeMol(gen_mol)
            Chem.AssignStereochemistry(gen_mol)
                    
        logging.info(f'{ligand_filename}: has attach: {sum(has_attach_l)}')
    
        return gen_mols
    
    
class YmirEarly(SBModel):
    
    def __init__(self,
                 minimized_path: str,
                 name: str = 'YmirEarly',
                 gen_path = '/home/bb596/hdd/ymir/generated_cross_docked_early/'
                 ) -> None:
        super().__init__(name=name,
                         minimized_path=minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        _, real_ligand_filename = ligand_filename.split('/')
        gen_mols_filename = real_ligand_filename.replace('.sdf', '.sdf_generated.sdf')
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in Ymir results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
            
        has_attach_l = []
        for gen_mol in gen_mols:
            has_attach = False
            for atom in gen_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    has_attach = True
            has_attach_l.append(has_attach)
            Chem.SanitizeMol(gen_mol)
            Chem.AssignStereochemistry(gen_mol)
                    
        logging.info(f'{ligand_filename}: has attach: {sum(has_attach_l)}')
    
        return gen_mols