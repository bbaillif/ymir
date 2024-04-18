import os
import copy

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Mol
from ymir.data import PDBbind
from ymir.utils.fragment import get_unique_fragments_from_mols, select_mol_with_symbols
from ymir.data import Fragment
from ymir.utils.mol_conversion import (rdkit_conf_to_ccdc_mol, 
                                       ccdc_mol_to_rdkit_mol)
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.Descriptors import MolWt
from ccdc.conformer import ConformerGenerator
from scipy.spatial.transform import Rotation
from ymir.utils.spatial import rotate_conformer, translate_conformer


class FragmentLibrary():
    
    def __init__(self,
                 ligands: list[Mol] = None,
                 fragments: list[Fragment] = None,
                 removeHs: bool = True,
                 subset: str = 'refined') -> None:
        self.pdbbind = PDBbind()
        assert subset in ['all', 'refined', 'general'], \
            'Select one PDBbind subset among all, refined or general'
        self.subset = subset
            
        self.pdbbind_ligand_path = f'/home/bb596/hdd/ymir/pdbbind_ligands_{self.subset}.sdf'
        self.fragments_path = f'/home/bb596/hdd/ymir/small_fragments_3D_{self.subset}.sdf' 
        self.removeHs = removeHs
        
        self._ligands = ligands # Default are PDBbind ligands, when ligands is None
        self._fragments = fragments # Default are constructed from PDBbind ligands, when fragments is None
        self._protected_fragments = None
        
    @property
    def ligands(self) -> list[Mol]:
        if self._ligands is None:
            self._ligands = self.get_pdbbind_ligands()
        return self._ligands
    
    
    @property
    def fragments(self) -> list[Fragment]:
        if self._fragments is None:
            self._fragments = self.get_fragments()
        return self._fragments
    
    
    @property
    def protected_fragments(self) -> list[Fragment]:
        if self._protected_fragments is None:
            self._protected_fragments = self.get_protected_fragments()
        return self._protected_fragments
        
        
    def get_pdbbind_ligands(self) -> list[Mol]:
        if not os.path.exists(self.pdbbind_ligand_path):
            self.write_pdbbind_ligands()

        assert os.path.exists(self.pdbbind_ligand_path)
        ligands = [mol for mol in Chem.SDMolSupplier(self.pdbbind_ligand_path)]
        
        return ligands
    
    
    def write_pdbbind_ligands(self) -> None:
        ligands = self.pdbbind.get_ligands(subset=self.subset)
        with Chem.SDWriter(self.pdbbind_ligand_path) as writer:
            for mol in ligands:
                # Set PDB ID as molecule name, there is a unique ligand for each PDB ID in PDBbind
                pdb_id = mol.GetConformer().GetProp('PDB_ID')
                mol.SetProp('PDB_ID', pdb_id)
                writer.write(mol)
                
                
    def get_fragments(self) -> list[Fragment]:
        if not os.path.exists(self.fragments_path):
            self.write_fragments()
               
        assert os.path.exists(self.fragments_path)
        small_frags_3D = [Fragment(mol) for mol in Chem.SDMolSupplier(self.fragments_path,
                                                                      removeHs=self.removeHs)]
        return small_frags_3D
    
    
    def select_small_fragments(self,
                               fragments: list[Fragment]) -> list[Fragment]:
        small_fragments = [frag for frag in fragments 
                            if (frag.GetNumAtoms() <=25) 
                            and (CalcNumRotatableBonds(frag) <= 3) 
                            and (MolWt(frag) <= 150)]
        return small_fragments


    def write_fragments(self) -> None:
        ligands_h = [Chem.AddHs(mol, addCoords=True) for mol in self.ligands]
        unique_fragments = get_unique_fragments_from_mols(mols=ligands_h)
        small_fragments = self.select_small_fragments(fragments=unique_fragments)

        # import pdb;pdb.set_trace()

        cg = ConformerGenerator()

        # small_frags_copy = [Fragment(f) for f in small_fragments]
        small_frags_3D: list[Fragment] = []
        for i, fragment in enumerate(tqdm(small_fragments)):

            f = Fragment(fragment)
            if len(f.get_attach_points()) == 0:
                import pdb;pdb.set_trace()
            f.protect()
            protections = f.protections

            ccdc_mol = rdkit_conf_to_ccdc_mol(rdkit_mol=f)
            conformer_hits = cg.generate(ccdc_mol)
            for conformer_hit in conformer_hits:
                conformer = conformer_hit.molecule
                rdkit_mol = ccdc_mol_to_rdkit_mol(conformer)
                # rdkit_mol = Chem.RemoveHs(rdkit_mol)
                # we cast to protected fragment to remove the protections next
                fragment3D = Fragment(rdkit_mol)
                fragment3D.unprotect(protections=protections)
                # import pdb;pdb.set_trace()
                small_frags_3D.append(fragment3D)

        for frag in small_frags_3D:
            frag.center()
            
        with Chem.SDWriter(self.fragments_path) as writer:
            for mol in small_frags_3D:
                writer.write(mol)


    def get_protected_fragments(self) -> list[Fragment]:
        protected_fragments = []
        for fragment in self.fragments:
            attach_points = fragment.get_attach_points()
            if not 7 in list(attach_points.values()): # "7 double bond 7" reaction does not work for some reason...
                for atom_id, label in attach_points.items():
                    fragment_copy = copy.deepcopy(fragment)
                    protected_fragment = Fragment(fragment_copy)
                    protected_fragment.protect(atom_ids_to_keep=[atom_id])
                    protected_fragments.append(protected_fragment)
        
        return protected_fragments
                

    def get_restricted_fragments(self,
                                 z_list: list[int],
                                 max_attach: int = 10):
        protected_fragments = select_mol_with_symbols(self.protected_fragments,
                                              z_list)

        # Select only fragments with at most max_attach point
        n_attaches = []
        for fragment in protected_fragments:
            frag_copy = Fragment(mol=fragment,
                                protections=fragment.protections)
            frag_copy.unprotect()
            attach_points = frag_copy.get_attach_points()
            n_attach = len(attach_points)
            n_attaches.append(n_attach)

        protected_fragments = [fragment 
                            for fragment, n in zip(protected_fragments, n_attaches)
                            if n <= max_attach]
                
        unique_fragments = []
        unique_smiles = []
        for mol in protected_fragments:
            smiles = Chem.MolToSmiles(mol)
            if not smiles in unique_smiles:
                unique_smiles.append(smiles)
                unique_fragments.append(mol)
                
        protected_fragments = unique_fragments
        
        return protected_fragments
    
    
    def get_restricted_ligands(self,
                               z_list: list[int]):
        ligands = [ligand 
                for ligand in self.ligands 
                if ligand.GetNumHeavyAtoms() < 50]

        # Remove ligands having at least one heavy atom not in list
        ligands = select_mol_with_symbols(ligands,
                                        z_list)
        
        return ligands