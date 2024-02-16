import time
import logging
import torch

from rdkit.Chem import Mol
from ymir.data.structure import Pocket, Complex
from ymir.metrics.activity import GlideScore, VinaScore
from ymir.metrics.steric_clash import StericClash
from typing import Union

class DockingBatchRewards():
    
    def __init__(self,
                 complx: Complex,
                 scorer: VinaScore,
                 max_num_heavy_atoms: int = 50,
                 clash_penalty: float = -100.0) -> None:
        self.complx = complx
        self.scorer = scorer
        self.clash_detector = StericClash()
        self.max_num_heavy_atoms = max_num_heavy_atoms
        self.clash_penalty = clash_penalty
        
    
    def get_rewards(self,
                    mols: list[Mol],
                    terminateds: list[bool]) -> list[float]:
        
        t0 = time.time()
        
        n_clashes = self.clash_detector.get(self.complx.pocket.mol, mols)
        mol_idxs_to_score = []
        mols_to_score = []
        rewards = [0] * len(mols) # Default is 0 : construction is ongoing, and no clash
        for mol_idx, (n_clash, mol, terminated) in enumerate(zip(n_clashes, mols, terminateds)):
            has_clash = n_clash > 0
            if has_clash: # clash is penalized
                rewards[mol_idx] = self.clash_penalty
            elif terminated: # we compute the docking score
                mol_idxs_to_score.append(mol_idx)
                mols_to_score.append(mol)
        
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
