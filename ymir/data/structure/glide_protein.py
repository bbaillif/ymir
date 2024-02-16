import os
import subprocess
import logging

from rdkit.Chem import Mol
from .protein import Protein
from ymir.params import SCHRODINGER_PATH, GLIDE_OUTPUT_DIRPATH


class GlideProtein(Protein):
    
    def __init__(self, 
                 pdb_filepath: str,
                 native_ligand: Mol,
                 glide_output_dirpath: str = GLIDE_OUTPUT_DIRPATH) -> None:
        super().__init__(pdb_filepath)
        self.glide_output_dirpath = glide_output_dirpath
        if not os.path.exists(self.glide_output_dirpath):
            os.mkdir(self.glide_output_dirpath)
        
        self.mae_filepath = pdb_filepath.replace('.pdb', 
                                                '.mae')
        if not os.path.exists(self.mae_filepath):
            self.generate_mae_file()
           
        assert os.path.exists(self.mae_filepath)
           
        self.grid_filepath = pdb_filepath.replace('.pdb', 
                                                  '.zip')
        self.grid_center = native_ligand.GetConformer().GetPositions().mean(axis=0)
            
        self.glide_grid_in_filename = 'glide_grid_generation.in'
        self.glide_grid_in_filepath = self.glide_grid_in_filename
        
        if not os.path.exists(self.grid_filepath):
            if os.path.exists(self.glide_grid_in_filepath):
                os.remove(self.glide_grid_in_filepath)
            self.generate_glide_grid_in_file(self.grid_center)
            assert os.path.exists(self.glide_grid_in_filepath)
            self.generate_grid_file()
            
        assert os.path.exists(self.grid_filepath)
        
    
    def generate_mae_file(self):
        logging.info(f'Converting {self.pdb_filepath} to {self.mae_filepath}')
        command = [f'{SCHRODINGER_PATH}/utilities/structconvert',
                   self.pdb_filepath,
                   self.mae_filepath]
        subprocess.run(command)
        
        
    def generate_glide_grid_in_file(self,
                                    grid_center: list[float]):
        # List of keywords available in the Glide documentation
        logging.info(f'Writing glide grid generation input in {self.glide_grid_in_filepath}')
        grid_center_str = [str(value) for value in grid_center]
        d = {'GRIDFILE': self.grid_filepath,
             'OUTPUTDIR': self.glide_output_dirpath,
             'RECEP_FILE': self.mae_filepath,
             'GRID_CENTER': ','.join(grid_center_str)}
        with open(self.glide_grid_in_filepath, 'w') as f:
            for param_name, value in d.items():
                f.write(f'{param_name}   {value}')
                f.write('\n')
        
        
    def generate_grid_file(self):
        logging.info(f'Generate Glide grid using {self.glide_grid_in_filepath}')
        glide_binpath = os.path.join(SCHRODINGER_PATH, 'glide')
        command = [f'{glide_binpath}',
                   self.glide_grid_in_filepath,
                   '-WAIT']
        subprocess.run(command)
        
        
    