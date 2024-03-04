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
        
        # To be loaded on demand
        self._vina_protein = None
        self._protein_clean = None
        self._pocket = None
        
        # self.clash_detector = StericClash(self.pocket)
        
        # self.vina_scorer = VinaScorer(vina_protein=self.vina_protein)
        # self.vina_scorer.set_box_from_ligand(self.ligand)
        # self.vina_score = VinaScore(vina_scorer=self.vina_scorer)
        
        # self.glide_protein = GlideProtein(pdb_filepath=self.vina_protein.protein_clean_filepath,
        #                             native_ligand=ligand)
        # featurizer = RDKitFeaturizer()
        # data_list = featurizer.featurize_mol(self.pocket.mol)
        # self.protein_pocket_data = data_list[0]
        
    @property
    def vina_protein(self) -> VinaProtein:
        if self._vina_protein is None:
            self._vina_protein = VinaProtein(pdb_filepath=self.protein_path)
        return self._vina_protein
        
    @property
    def protein_clean(self) -> Protein:
        if self._protein_clean is None:
            self._protein_clean = Protein(self.vina_protein.protein_clean_filepath)
        return self._protein_clean
        
    @property
    def pocket(self) -> Pocket:
        pocket = Pocket(protein=self.protein_clean, 
                        native_ligand=self.ligand)
        return pocket
        # if self._pocket is None:
        #     self._pocket = Pocket(protein=self.protein_clean, 
        #                             native_ligand=self.ligand)
        # return self._pocket
    
    