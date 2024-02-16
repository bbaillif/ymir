import time
import logging
import torch

from rdkit.Chem import Mol
from ymir.data.structure import Pocket
from ymir.metrics.activity import GlideScore, VinaScore
from ymir.metrics.steric_clash import StericClash
from typing import Union

class DockingBatchRewards():
    
    def __init__(self,
                 scorer: Union[GlideScore, VinaScore],
                 pocket: Pocket,
                 max_num_heavy_atoms: int = 50) -> None:
        self.scorer = scorer
        self.pocket = pocket
        self.clash_detector = StericClash(self.pocket)
        self.max_num_heavy_atoms = max_num_heavy_atoms
        
    
    def get_rewards(self,
                    mols: list[Mol]) -> list[float]:
        
        t0 = time.time()
        
        n_clashes = self.clash_detector.get(mols)
        mol_idxs_to_score = []
        mols_to_score = []
        for mol_idx, (n_clash, mol) in enumerate(zip(n_clashes, mols)):
            if n_clash == 0:
                mol_idxs_to_score.append(mol_idx)
                mols_to_score.append(mol)
        
        rewards = [-100] * len(mols) # Default: mol has a clash
        
        if len(mols_to_score) > 0:
        
            scores = self.scorer.get(mols_to_score)
            
            for mol_idx, score in zip(mol_idxs_to_score, scores):
                rewards[mol_idx] = -score # we want to maximize the reward, so minimize glide score
        
        for i, (reward, mol) in enumerate(zip(rewards, mols)):
            n_atoms = mol.GetNumHeavyAtoms()
            if n_atoms > self.max_num_heavy_atoms:
                malus = n_atoms - self.max_num_heavy_atoms
                malus_reward = reward - malus
                rewards[i] = malus_reward
        
        # rewards = torch.tensor(rewards)
        
        # include
        
        # scores = torch.Tensor(scores)
        # rewards = - scores # we want to maximize the reward, so minimize glide score
        # / np.sqrt(self.seed.GetNumAtoms())
        # INCLUDE TORSION PREFERENCE PENALTY ?
        t1 = time.time()
        logging.info(f'Time for reward: {t1 - t0}')
        return rewards
