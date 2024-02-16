from rdkit.Chem import Mol
from ymir.data.structure import VinaProtein, Protein, Pocket, GlideProtein
from ymir.metrics.activity import VinaScorer, VinaScore
# from ymir.data.featurizer import RDKitFeaturizer
from ymir.metrics.steric_clash import StericClash

class Complex():
    
    def __init__(self,
                 ligand: Mol,
                 protein_path: str) -> None:
        self.ligand = ligand
        self.protein_path = protein_path
        self.vina_protein = VinaProtein(pdb_filepath=protein_path)
        self.protein_clean = Protein(self.vina_protein.protein_clean_filepath)
        self.pocket = Pocket(protein=self.protein_clean, 
                            native_ligand=ligand)
        # self.clash_detector = StericClash(self.pocket)
        
        # self.vina_scorer = VinaScorer(vina_protein=self.vina_protein)
        # self.vina_scorer.set_box_from_ligand(self.ligand)
        # self.vina_score = VinaScore(vina_scorer=self.vina_scorer)
        
        # self.glide_protein = GlideProtein(pdb_filepath=self.vina_protein.protein_clean_filepath,
        #                             native_ligand=ligand)
        # featurizer = RDKitFeaturizer()
        # data_list = featurizer.featurize_mol(self.pocket.mol)
        # self.protein_pocket_data = data_list[0]