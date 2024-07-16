import os
import copy
import random

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
from ccdc.io import Molecule


class FragmentLibrary():
    
    def __init__(self,
                 ligands: list[Mol] = None,
                 fragments: list[Fragment] = None,
                 removeHs: bool = True,
                 subset: str = 'refined') -> None:
        self.pdbbind = PDBbind()
        assert subset in ['all', 'refined', 'general', 'cross_docked'], \
            'Select one PDBbind subset among all, refined or general'
        self.subset = subset
            
        self.pdbbind_ligand_path = f'/home/bb596/hdd/ymir/{self.subset}_ligands.sdf'
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
        ligands = [mol for mol in Chem.SDMolSupplier(self.pdbbind_ligand_path, removeHs=self.removeHs)]
        
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
                            if (frag.mol.GetNumAtoms() <= 25) 
                            and (CalcNumRotatableBonds(frag.mol) <= 3) 
                            and (MolWt(frag.mol) <= 150)]
        return small_fragments
    
    
    def select_single_bond_fragments(self,
                                     fragments: list[Fragment]) -> list[Fragment]:
        single_bond_fragments = [frag 
                                 for frag in fragments 
                                 if not 7 in list(frag.get_attach_points().values())]
        return single_bond_fragments


    def get_unique_mols(self,
                        mols: list[Mol]) -> list[Mol]:
        unique_mols = []
        unique_smiles = []
        for mol in mols:
            smiles = Chem.MolToSmiles(mol)
            if not smiles in unique_smiles:
                unique_smiles.append(smiles)
                unique_mols.append(mol)
        return unique_mols


    def write_fragments(self,
                        unique_mols: bool = True) -> None:
        
        ligands = self.ligands
        if unique_mols:
            mols = self.get_unique_mols(mols=ligands)
        else:
            mols = ligands
        mols_h = [Chem.AddHs(mol, addCoords=True) for mol in mols]
        unique_fragments = get_unique_fragments_from_mols(mols=mols_h)
        small_fragments = self.select_small_fragments(fragments=unique_fragments)
        small_fragments = self.select_single_bond_fragments(fragments=small_fragments)

        # import pdb;pdb.set_trace()

        cg = ConformerGenerator()

        small_frags_3D: list[Fragment] = []
        for i, fragment in enumerate(tqdm(small_fragments)):

            f = Fragment.from_fragment(fragment)
            if len(f.get_attach_points()) == 0:
                import pdb;pdb.set_trace()
            f.protect(protection_atomic_num=6)
            protections = f.protections

            mol = Mol(f.mol)
            Chem.SanitizeMol(mol)
            # Adds hydrogens to carbon protections
            mol = Chem.AddHs(mol, addCoords=True)
            
            if mol.GetNumAtoms() != f.mol.GetNumAtoms() + 3 * len(protections): # 3 hydrogens added
                import pdb;pdb.set_trace()
            
            try:
                ccdc_mol = rdkit_conf_to_ccdc_mol(rdkit_mol=mol)
            except:
                import pdb;pdb.set_trace()
            conformer_hits = cg.generate(ccdc_mol)
            for conformer_hit in conformer_hits:
                conformer = conformer_hit.molecule
                # ccdc_mol = Molecule.from_string(conformer.to_string())
                # ccdc_mol.remove_atoms([atom 
                #                         for atom in ccdc_mol.atoms
                #                         if atom.atomic_number <= 1])
                rdkit_mol = ccdc_mol_to_rdkit_mol(conformer)
                # rdkit_mol = Chem.RemoveHs(rdkit_mol)
                # we cast to protected fragment to remove the protections next
                fragment3D = Fragment(rdkit_mol)
                # import pdb;pdb.set_trace()
                fragment3D.unprotect(protections=protections)
                # Chem.SanitizeMol(fragment3D.mol)
                
                small_frags_3D.append(fragment3D)

        for frag in small_frags_3D:
            frag.center()
            
        with Chem.SDWriter(self.fragments_path) as writer:
            for frag in small_frags_3D:
                mol = frag.mol
                writer.write(mol)
        
        # import pdb;pdb.set_trace()


    def get_protected_fragments(self) -> list[Fragment]:
        protected_fragments = []
        smiles_combinations = []
        for fragment in self.fragments:
            attach_points = fragment.get_attach_points()
            if not 7 in list(attach_points.values()): # "7 double bond 7" reaction does not work for some reason...
                for atom_id, label in attach_points.items():
                    protected_fragment = Fragment.from_fragment(fragment)
                    protected_fragment.protect(atom_ids_to_keep=[atom_id])
                    protected_smiles = Chem.MolToSmiles(protected_fragment.mol)
                    unprotected_smiles = Chem.MolToSmiles(fragment.mol)
                    combo = [protected_smiles, unprotected_smiles]
                    if not combo in smiles_combinations:
                        protected_fragments.append(protected_fragment)
                        smiles_combinations.append(combo)
        
        return protected_fragments
                

    def get_restricted_fragments(self,
                                 z_list: list[int],
                                 max_attach: int = 10,
                                 max_torsions: int = 1,
                                 n_fragments: int = None,
                                 get_unique: bool = False,
                                 shuffle: bool = False,
                                 remove_hs: bool = True) -> dict[str, tuple[Fragment, list[int]]]:
        
        fragments = copy.deepcopy(self.fragments)
        if shuffle:
            random.shuffle(fragments)
            
        if remove_hs:
            for fragment in fragments:
                fragment.remove_hs()
            
        fragments = select_mol_with_symbols(fragments,
                                            z_list)
        
        # Select only fragments with at most max_attach point
        n_attaches = []
        for fragment in fragments:
            attach_points = fragment.get_attach_points()
            n_attach = len(attach_points)
            n_attaches.append(n_attach)

        fragments = [fragment 
                    for fragment, n in zip(fragments, n_attaches)
                    if n <= max_attach]
            
        n_torsions = [CalcNumRotatableBonds(frag.mol) 
                      for frag in fragments]
        fragments = [frag 
                    for frag, n_torsion in zip(fragments, n_torsions) 
                    if n_torsion <= max_torsions]
            
        if get_unique:
            unique_fragments = []
            unique_smiles = []
            for frag in fragments:
                mol = frag.mol
                up_smiles = Chem.MolToSmiles(mol)
                if not up_smiles in unique_smiles:
                    unique_smiles.append(up_smiles)
                    unique_fragments.append(frag)
                    
            fragments = unique_fragments
            
        protected_fragments: dict[str, list] = {}
        for fragment in fragments:
            attach_points = fragment.get_attach_points()
            if not 7 in list(attach_points.values()): # "7 double bond 7" reaction does not work for some reason...
                for atom_id, label in attach_points.items():
                    protected_fragment = Fragment.from_fragment(fragment)
                    protected_fragment.protect(atom_ids_to_keep=[atom_id])
                    
                    reverse_protected_fragment = Fragment.from_fragment(fragment)
                    atom_ids_to_keep = list(range(0, fragment.mol.GetNumAtoms()))
                    atom_ids_to_keep.remove(atom_id)
                    reverse_protected_fragment.protect(atom_ids_to_keep=atom_ids_to_keep)
                    
                    rev_protected_smiles = Chem.MolToSmiles(reverse_protected_fragment.mol)
                    if not rev_protected_smiles in protected_fragments:
                        protected_fragments[rev_protected_smiles] = [protected_fragment, [label]]
                    else:
                        if not label in protected_fragments[rev_protected_smiles][1]:
                            protected_fragments[rev_protected_smiles][1].append(label)
            
        if n_fragments is not None:
            protected_fragments = {k: v for k, v in list(protected_fragments.items())[:n_fragments]}
            
        return protected_fragments


    # def get_restricted_fragments(self,
    #                              z_list: list[int],
    #                              max_attach: int = 10,
    #                              max_torsions: int = 1,
    #                              n_fragments: int = None,
    #                              get_unique: bool = False,
    #                              shuffle: bool = False) -> list[Fragment]:
        
    #     if shuffle:
    #         protected_fragments = copy.deepcopy(self.protected_fragments)
    #         random.shuffle(protected_fragments)
        
    #     protected_fragments = select_mol_with_symbols(self.protected_fragments,
    #                                           z_list)

    #     # Select only fragments with at most max_attach point
    #     n_attaches = []
    #     for fragment in protected_fragments:
    #         frag_copy = Fragment.from_fragment(fragment)
    #         frag_copy.unprotect()
    #         attach_points = frag_copy.get_attach_points()
    #         n_attach = len(attach_points)
    #         n_attaches.append(n_attach)

    #     protected_fragments = [fragment 
    #                         for fragment, n in zip(protected_fragments, n_attaches)
    #                         if n <= max_attach]
                
    #     if get_unique:
    #         unique_fragments = []
    #         unique_combos = []
    #         for frag in protected_fragments:
    #             mol = frag.mol
    #             p_smiles = Chem.MolToSmiles(mol)
    #             frag_copy = Fragment.from_fragment(frag)
    #             frag_copy.unprotect()
    #             up_smiles = Chem.MolToSmiles(frag_copy.mol)
    #             combo = (p_smiles, up_smiles)
    #             if not combo in unique_combos:
    #                 unique_combos.append(combo)
    #                 unique_fragments.append(frag)
                    
    #         protected_fragments = unique_fragments
            
    #     n_torsions = [CalcNumRotatableBonds(frag.mol) for frag in protected_fragments]
    #     protected_fragments = [frag for frag, n_torsion in zip(protected_fragments, n_torsions) if n_torsion <= max_torsions]
            
    #     if n_fragments is not None:
    #         protected_fragments = protected_fragments[:n_fragments]
            
    #     return protected_fragments
    
    
    def get_restricted_ligands(self,
                               z_list: list[int]):
        ligands = [ligand 
                for ligand in self.ligands 
                if ligand.GetNumHeavyAtoms() < 50]

        # Remove ligands having at least one heavy atom not in list
        ligands = select_mol_with_symbols(ligands,
                                        z_list)
        
        # ligands_h = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]
        
        return ligands