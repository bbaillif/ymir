import numpy as np
import logging
import time

from rdkit.Chem import Mol
from .vina_scorer import VinaScorer

class VinaScore():
    
    def __init__(self, 
                 vina_scorer: VinaScorer,
                 name: str = 'Vina score',
                 minimized: bool = False,
                 ) -> None:
        self.scores = {}
        self.vina_scorer = vina_scorer
        self.minimized = minimized
        
        
    # def get(self, 
    #         cel: GeneratedCEL) -> float:
    #     all_scores = []
    #     mols = cel.to_mol_list()
    #     for mol in mols:
    #         try:
    #             # start_time = time.time()
    #             all_scores = self.vina_scorer.score_mols(ligands=mols, 
    #                                                 minimized=self.minimized)
    #             if all_scores is None:
    #                 raise Exception('Failed molecule preparation')
    #             # logging.info(f'Time: {time.time() - start_time}')
    #         except Exception as e:
    #             logging.warning(f'Vina scoring error: {e}')
            
    #     if len(all_scores) != cel.n_total_confs:
    #         import pdb;pdb.set_trace()
        
    #     return all_scores
        
        
        
    def get(self, 
            mols: list[Mol],
            add_hydrogens: bool = False) -> float:
        all_scores = []
        for mol in mols:
            try:
                # start_time = time.time()
                scores = self.vina_scorer.score_mol(ligand=mol, 
                                                    minimized=self.minimized,
                                                    add_hydrogens=add_hydrogens)
                if scores is None:
                    raise Exception('Failed molecule preparation')
                # logging.info(f'Time: {time.time() - start_time}')
                score = scores[0]
                all_scores.append(score)
            except Exception as e:
                logging.warning(f'Vina scoring error: {e}')
                all_scores.append(100)
            
        assert len(all_scores) == len(mols)
        
        return all_scores
            