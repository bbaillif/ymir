import os
import glob
import subprocess
import logging

from rdkit import Chem
from rdkit.Chem import Mol
from meeko import (MoleculePreparation,
                   PDBQTWriterLegacy)
from ymir.params import (VINA_LIGANDS_DIRECTORY,
                         VINA_OUTPUT_DIRECTORY,
                         VINA_CONFIG_PATH,
                         VINA_CPUS,
                         VINA_SEED,
                         VINA_PATH)
from multiprocessing import Pool, TimeoutError

def run_vina(vina_cmd: list[str]):
    score = 0
    try:
        vina_cmd = ' '.join(vina_cmd)
        completed_process = subprocess.run(vina_cmd, capture_output=True, shell=True, timeout=60)
        stdout = completed_process.stdout.decode('utf-8')
        stderr = completed_process.stderr.decode('utf-8')
        logging.info(stdout)
        if len(stderr) > 2:
            logging.info(stderr)
        
        for line in stdout.split('\n'):
            if line.startswith('Estimated Free Energy of Binding   :'):
                score = line.split(':')[1].split('(')[0].strip()
                score = float(score)
    except subprocess.TimeoutExpired:
        logging.info('Subprocess timeout')
    return score

def fake_fn(vina_cmd: str) -> str:
    return vina_cmd

class VinaCLI():
    
    def __init__(self,
                 score_only: bool = True,
                 ligands_directory: str = VINA_LIGANDS_DIRECTORY,
                 output_directory: str = VINA_OUTPUT_DIRECTORY,
                 n_threads: int = VINA_CPUS,
                 seed: int = VINA_SEED,
                 config_path: str = VINA_CONFIG_PATH,
                 vina_path: str = VINA_PATH,
                 verbosity: int = 2) -> None:
        self.score_only = score_only
        self.ligands_directory = ligands_directory
        self.output_directory = output_directory
        self.n_threads = n_threads
        self.seed = seed
        self.config_path = config_path
        self.vina_path = vina_path
        self.verbosity = verbosity
        self.base_vina_cmd = self.get_base_vina_cmd()
        
    def get(self,
            receptor_paths: list[str],
            native_ligands: list[Mol],
            ligands: list[Mol],
            add_hydrogens: bool = False):
        
        for path in receptor_paths:
            assert path.endswith('.pdbqt')
        assert len(receptor_paths) == len(native_ligands)
        assert len(receptor_paths) == len(ligands)
        
        if add_hydrogens:
            ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]
            
        old_ligand_files = glob.glob(os.path.join(self.ligands_directory, '*'))
        for f in old_ligand_files:
            os.remove(f)
            
        old_poses_files = glob.glob(os.path.join(self.output_directory, '*'))
        for f in old_poses_files:
            os.remove(f)
            
        ligand_paths = self.write_ligands(ligands)
        base_vina_cmd = self.get_base_vina_cmd()
        
        vina_cmds = []
        for ligand_path, receptor_path, native_ligand in zip(ligand_paths, receptor_paths, native_ligands):
            vina_cmd = self.get_ligand_vina_cmd(base_vina_cmd, 
                                                receptor_path, 
                                                native_ligand,
                                                ligand_path)
            vina_cmds.append(vina_cmd)
        
        logging.info('Run Vina CLI')
        scores = []
        try:
            with Pool(self.n_threads, maxtasksperchild=1) as pool:
                results = pool.map_async(run_vina, vina_cmds)
                for score in results.get(timeout=10*len(ligands)):
                    scores.append(score)
                pool.close()
                pool.join()
        except TimeoutError:
            logging.info('Pool Timeout')
            
        # for vina_cmd in vina_cmds:
        #     score = run_vina(vina_cmd)
        #     scores.append(score)
        # try:
        #     with Pool(self.n_threads, maxtasksperchild=1) as pool:
        #         results = pool.map_async(fake_fn, vina_cmds)
        #         for r in results.get(timeout=2):
        #             logging.info(r)
        #         pool.close()
        #         pool.join()
        # except TimeoutError:
        #     logging.info('Pool Timeout')
        
        return scores
            
    def write_ligands(self,
                      ligands: list[Mol]) -> list[str]: 
        
        ligand_pathes = []
        for i, ligand in enumerate(ligands) :
            ligand_name = f'Ligand_{i}'
            preparator = MoleculePreparation(merge_these_atom_types=(),
                                             rigid_macrocycles=True)
            mol_setups = preparator.prepare(ligand)
            mol_setup = mol_setups[0]
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup=mol_setup, 
                                                                            add_index_map=True,
                                                                            remove_smiles=True,
                                                                            bad_charge_ok=True)
            if 'G' in pdbqt_string:
                import pdb;pdb.set_trace()
                
            if is_ok:
                ligand_path = os.path.join(self.ligands_directory, f'{ligand_name}.pdbqt')
                ligand_pathes.append(ligand_path)
                with open(ligand_path, 'w') as f:
                    f.write(pdbqt_string)
            else:
                import pdb;pdb.set_trace()
        
        return ligand_pathes
    
    
    def get_base_vina_cmd(self):
        
        vina_cmd = [self.vina_path,
                    # '--config', self.config_path, # might not work with Vina 1.2.5
                    '--cpu', str(1),
                    '--local_only',
                    '--verbosity', str(self.verbosity)]
        
        return vina_cmd
    
    
    def get_ligand_vina_cmd(self,
                            base_vina_cmd: list[str],
                            receptor_path: str,
                            native_ligand: Mol,
                            ligand_path: str,
                            box_padding: float = 15.0):
        ligand_positions = native_ligand.GetConformer().GetPositions()
        center = (ligand_positions.max(axis=0) + ligand_positions.min(axis=0)) / 2
        
        ligand_size = ligand_positions.max(axis=0) - ligand_positions.min(axis=0)
        box_size = ligand_size + box_padding
        
        output_path = os.path.join(self.output_directory, 
                                   ligand_path.split('/')[-1].replace('.pdbqt', '_out.pdbqt'))
        vina_cmd = base_vina_cmd + ['--ligand', ligand_path,
                                    '--out', output_path,
                                    '--receptor', receptor_path,
                                    '--center_x', f'{center[0]:.3f}',
                                    '--center_y', f'{center[1]:.3f}',
                                    '--center_z', f'{center[2]:.3f}',
                                    '--size_x', f'{box_size[0]:.3f}',
                                    '--size_y', f'{box_size[1]:.3f}',
                                    '--size_z', f'{box_size[2]:.3f}',]
        return vina_cmd
        
        
    # def set_config(self,
    #                receptor_path: str,
    #                native_ligand: Mol,
    #                box_padding: float = 5 # 5 to acccount for large fragment addition
    #                ):
    #     config_lines = []
    #     config_lines.append(f'receptor = {receptor_path}')
    #     config_lines.append(f'dir = {self.output_directory}')
    #     config_lines.append(f'cpu = {self.n_cpus}')
        
    #     # config_lines.append(f'autobox') # doesnt work
    #     # ligand_positions = native_ligand.GetConformer().GetPositions()
    #     # center = (ligand_positions.max(axis=0) + ligand_positions.min(axis=0)) / 2
    #     # config_lines.append(f'center_x = {center[0]}')
    #     # config_lines.append(f'center_y = {center[1]}')
    #     # config_lines.append(f'center_z = {center[2]}')
        
    #     # ligand_size = ligand_positions.max(axis=0) - ligand_positions.min(axis=0)
    #     # box_size = ligand_size + box_padding
    #     # config_lines.append(f'size_x = {box_size[0]}')
    #     # config_lines.append(f'size_y = {box_size[1]}')
    #     # config_lines.append(f'size_z = {box_size[2]}')

    #     with open(self.config_path, 'w') as f:
    #         f.write('\n'.join(config_lines))