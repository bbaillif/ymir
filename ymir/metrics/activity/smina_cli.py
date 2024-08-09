import os
import numpy as np
import glob
import subprocess
import logging

from rdkit import Chem
from rdkit.Chem import Mol
from meeko import (MoleculePreparation,
                   PDBQTWriterLegacy)
from ymir.params import (SMINA_LIGANDS_DIRECTORY,
                         SMINA_OUTPUT_DIRECTORY,
                         SMINA_CONFIG_PATH,
                         SMINA_CPUS,
                         SMINA_SEED,
                         SMINA_PATH,
                         SMINA_MAPS_DIRECTORY)
from multiprocessing import Pool, TimeoutError
from rdkit.Chem import AllChem
from ymir.data import Fragment
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from rdkit.Chem.rdFMCS import FindMCS
from scipy.spatial.distance import euclidean

class SminaCommand():
    
    def __init__(self,
                 smina_cmd: list[str],
                 receptor_path: str,
                 maps_path: str= SMINA_MAPS_DIRECTORY) -> None:
        self.smina_cmd = smina_cmd
        self.receptor_path = receptor_path
        self.maps_path = maps_path
        
        
    def to_receptor(self):
        
        new_smina_cmd = list(self.smina_cmd)
        
        new_smina_cmd.append('--receptor')
        new_smina_cmd.append(self.receptor_path)
        
        return new_smina_cmd
        
    def to_map_writing(self):
        
        new_smina_cmd = list(self.smina_cmd)
        
        new_smina_cmd.append('--receptor')
        new_smina_cmd.append(self.receptor_path)
        
        receptor_filename = self.receptor_path.split('/')[-1].replace('.pdbqt', '')
        receptor_maps_dirpath = os.path.join(self.maps_path, receptor_filename)
        new_smina_cmd.append('--write_maps')
        new_smina_cmd.append(receptor_maps_dirpath)
        
        new_smina_cmd.append('--force_even_voxels')
        
        return new_smina_cmd
    
    
    def to_map_loading(self):
        
        new_smina_cmd = list(self.smina_cmd)
        
        receptor_filename = self.receptor_path.split('/')[-1].replace('.pdbqt', '')
        receptor_maps_dirpath = os.path.join(self.maps_path, receptor_filename)
        new_smina_cmd.append('--maps')
        new_smina_cmd.append(receptor_maps_dirpath)
        
        return new_smina_cmd
               

def run_smina(smina_command: SminaCommand) -> list[float]:
    score = 0
    try:
        smina_cmd = smina_command.to_receptor()
        smina_cmd = ' '.join(smina_cmd)
        completed_process = subprocess.run(smina_cmd, capture_output=True, shell=True, timeout=60)
        stdout = completed_process.stdout.decode('utf-8')
        stderr = completed_process.stderr.decode('utf-8')
        if len(stderr) > 2:
            logging.info(stderr)
        
        scores = []
        for line in stdout.split('\n'):
            if line.startswith('Affinity:'):
                score = line.split(': ')[1].split(' ')[0].strip()
                score = float(score)
                scores.append(score)
    except subprocess.TimeoutExpired:
        logging.info('Subprocess timeout')
        
    return scores

def fake_fn(smina_cmd: str) -> str:
    return smina_cmd

class SminaCLI():
    
    def __init__(self,
                 score_only: bool = True,
                 ligands_directory: str = SMINA_LIGANDS_DIRECTORY,
                 output_directory: str = SMINA_OUTPUT_DIRECTORY,
                 n_threads: int = SMINA_CPUS,
                 seed: int = SMINA_SEED,
                 config_path: str = SMINA_CONFIG_PATH,
                 smina_path: str = SMINA_PATH,
                 verbosity: int = 0,
                 scoring_function: str = 'vinardo') -> None:
        self.score_only = score_only
        self.ligands_directory = ligands_directory
        self.output_directory = output_directory
        self.n_threads = n_threads
        self.seed = seed
        self.config_path = config_path
        self.smina_path = smina_path
        self.verbosity = verbosity
        self.scoring_function = scoring_function
        self.base_smina_cmd = self.get_base_smina_cmd()
        self.ligand_paths = None
        self.pose_paths = None
        
    def get(self,
            receptor_paths: list[str],
            ligands: list[Mol],
            add_hydrogens: bool = False):
        
        for path in receptor_paths:
            assert path.endswith('.pdbqt')
        assert len(receptor_paths) == len(ligands)
        
        # assert len(set(receptor_paths)) == 1
        
        if add_hydrogens:
            ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]
            
        old_ligand_files = glob.glob(os.path.join(self.ligands_directory, '*'))
        for f in old_ligand_files:
            os.remove(f)
            
        old_poses_files = glob.glob(os.path.join(self.output_directory, '*'))
        for f in old_poses_files:
            os.remove(f)
            
        receptor_to_ligand_i = {}
        receptor_ligands = {}
        for ligand_i, (ligand, receptor) in enumerate(zip(ligands, receptor_paths)):
            if receptor in receptor_ligands:
                receptor_ligands[receptor].append(ligand)
                receptor_to_ligand_i[receptor].append(ligand_i)
            else:
                receptor_ligands[receptor] = [ligand]
                receptor_to_ligand_i[receptor] = [ligand_i]
        n_receptors = len(receptor_ligands)
            
        self.write_ligands(receptor_ligands)
        
        new2old_ligand_i = []
        for receptor_path, ligand_idxs in receptor_to_ligand_i.items():
            new2old_ligand_i.extend(ligand_idxs)
            
        self.new2old = new2old_ligand_i
        self.old2new = np.argsort(new2old_ligand_i)
        
        base_smina_cmd = self.get_base_smina_cmd()
        
        if len(receptor_ligands) == 1:
            receptor_paths = [receptor_paths[0] for _ in self.ligand_paths]
        else:
            receptor_paths = list(receptor_ligands.keys())
        smina_commands = []
        for ligand_path, receptor_path in zip(self.ligand_paths, receptor_paths):
            smina_command = self.get_ligand_smina_cmd(base_smina_cmd, 
                                                    receptor_path, 
                                                    ligand_path)
            smina_commands.append(smina_command)
        
        logging.info('Run Smina CLI')
        all_scores = []
        if len(receptor_ligands) == 1 or self.n_threads == 1:
            for smina_command in smina_commands:
                scores = run_smina(smina_command)
                all_scores.extend(scores)
        else:
            try:
                with Pool(self.n_threads, maxtasksperchild=1) as pool:
                    results = pool.map_async(run_smina, smina_commands)
                    for scores in results.get(timeout=10*len(ligands)):
                        all_scores.extend(scores)
                    pool.close()
                    pool.join()
            except TimeoutError:
                logging.info('Pool Timeout')
            
        logging.info('Smina CLI finished')
        
        if len(all_scores) != len(ligands):
            import pdb;pdb.set_trace()
        
        all_scores = [all_scores[i] for i in self.old2new]
        
        return all_scores
            
    def write_ligands(self,
                      receptor_ligands: dict[str, list[Mol]],) -> list[str]: 
        
        # if len(ligands) < self.n_threads:
        #     batches = [ligands]
        # else:
        #     n_ligands_per_batch = len(ligands) // self.n_threads
        #     # n_ligands_per_batch = 1
        #     batches = [ligands[i:i + n_ligands_per_batch] for i in range(0, len(ligands), n_ligands_per_batch)]
            
        # batches = [ligands]
        # batches = [[ligand] for ligand in ligands]
           
        ligand_paths = []
            
        # if len(receptor_ligands) == 1 and len(receptor_ligands[list(receptor_ligands.keys())[0]]) >= self.n_threads:
        #     ligands = receptor_ligands[list(receptor_ligands.keys())[0]]
        #     batch_size = (len(ligands) // self.n_threads)
        #     modulo = len(ligands) % self.n_threads
        #     start = 0 
        #     for i in range(self.n_threads):
        #         if i < modulo:
        #             current_batch_size = batch_size + 1
        #         else:
        #             current_batch_size = batch_size
        #         end = start + current_batch_size
        #         batch = ligands[start:end]
        #         ligand_path = os.path.join(self.ligands_directory, f'ligands_batch_{i}.sdf')
        #         with Chem.SDWriter(ligand_path) as writer:
        #             for ligand in batch:
        #                 writer.write(ligand)
        #         ligand_paths.append(ligand_path)
        #         start = end
            
        # else:
        for i, receptor in enumerate(receptor_ligands):
            ligands = receptor_ligands[receptor]
            ligand_path = os.path.join(self.ligands_directory, f'ligands_batch_{i}.sdf')
            with Chem.SDWriter(ligand_path) as writer:
                for ligand in ligands:
                    writer.write(ligand)
            ligand_paths.append(ligand_path)
        
        # import pdb;pdb.set_trace()
        
        self.ligand_paths = ligand_paths
        self.pose_paths = []
    
    
    def get_base_smina_cmd(self):
        
        smina_cmd = [self.smina_path,
                    '--cpu', str(1),
                    '--verbosity', str(self.verbosity),
                    '--scoring', self.scoring_function]
        
        if self.score_only:
            smina_cmd.append('--score_only')
        else:
            smina_cmd.append('--minimize')
            smina_cmd.append('--minimize_early_term')
        
        return smina_cmd
    
    
    def get_ligand_smina_cmd(self,
                            base_smina_cmd: list[str],
                            receptor_path: str,
                            ligand_path: str) -> SminaCommand:
        
        receptor_filename = receptor_path.split('/')[-1].replace('.pdbqt', '')
        receptor_maps_dirpath = os.path.join(SMINA_MAPS_DIRECTORY, receptor_filename)
        
        output_path = os.path.join(self.output_directory, 
                                   ligand_path.split('/')[-1].replace('.sdf', '_out.sdf'))
        smina_cmd = base_smina_cmd + ['--ligand', ligand_path,
                                    '--out', output_path,
                                    ]
        self.pose_paths.append(output_path)
        smina_command = SminaCommand(smina_cmd, receptor_path)
        return smina_command
        
        
    def get_poses(self):
        all_poses = []
        for i, pose_path in enumerate(self.pose_paths):
            poses = [mol for mol in Chem.SDMolSupplier(pose_path)]
            assert all([mol is not None for mol in poses])
            all_poses.extend(poses)
            
        # for i, product in enumerate(products):
        #     pose_path = os.path.join(self.output_directory, f'Ligand_{i}_out.pdbqt')
        #     xyz_block = self.reader.to_xyz_block(pose_path)
        #     pose_mol = Chem.MolFromXYZBlock(xyz_block)
        #     if pose_mol is None:
        #         import pdb;pdb.set_trace()
        #     poses.append(pose_mol)

        all_poses = [all_poses[i] for i in self.old2new]

        return all_poses
        
        
    # def get_poses(self,
    #               products: list[Fragment],):
    #     poses = []
    #     for i, product in enumerate(products):
    #         pose_path = os.path.join(self.output_directory, f'Ligand_{i}_out.pdbqt')
    #         xyz_block = self.reader.to_xyz_block(pose_path)
    #         pose_mol = Chem.MolFromXYZBlock(xyz_block)
    #         if pose_mol is None:
    #             import pdb;pdb.set_trace()
    #         rdDetermineBonds.DetermineConnectivity(pose_mol)
    #         seed_copy = Fragment.from_fragment(product)
    #         seed_copy.protect()
    #         pose_mol = AllChem.AssignBondOrdersFromTemplate(seed_copy, pose_mol)
    #         Chem.SanitizeMol(pose_mol)
    #         Chem.SanitizeMol(seed_copy)
    #         mcs = FindMCS([seed_copy, pose_mol])
    #         if mcs.numAtoms != seed_copy.GetNumAtoms():
    #             import pdb;pdb.set_trace()
    #         seed_match = seed_copy.GetSubstructMatch(pose_mol)
    #         # print('Seed', seed_match)
    #         # print('Pose', pose_mol.GetSubstructMatch(seed_copy))
    #         if len(seed_match) != mcs.numAtoms:
    #             import pdb;pdb.set_trace()
            
    #         seed_conf = seed_copy.GetConformer()
    #         pose_coords = pose_mol.GetConformer().GetPositions()
    #         for pose_i, seed_i  in enumerate(seed_match):
    #             coord = pose_coords[pose_i]
    #             seed_symbol = seed_copy.GetAtomWithIdx(seed_i).GetSymbol()
    #             pose_symbol = pose_mol.GetAtomWithIdx(pose_i).GetSymbol()
    #             if seed_symbol != pose_symbol :
    #                 import pdb;pdb.set_trace()
    #             point = Point3D(*coord)
    #             seed_conf.SetAtomPosition(seed_i, point)
                
    #         ligand_centroid = product.mol.GetConformer().GetPositions().mean(0)
    #         pose_centroid = seed_conf.GetPositions().mean(0)
    #         centroid_distance = euclidean(ligand_centroid, pose_centroid)
    #         if centroid_distance > 5:
    #             import pdb;pdb.set_trace()
                
    #         poses.append(seed_copy)
            
    #     return poses
        
        
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