from .fragment import Fragment
from .pdbbind import PDBbind
from .structure import Complex
from .cross_docked import CrossDocked

import os
from rdkit import Chem
from ymir.utils.fragment import get_fragments_from_mol
from ymir.data import Fragment, Complex, CrossDocked
from ymir.params import CROSSDOCKED_POCKET10_PATH


def get_seeds_complx(params) -> tuple[list[Fragment], Complex]:
    protein_path, pocket_path, ligand = params
    fragments, frags_mol_atom_mapping = get_fragments_from_mol(ligand)
        
    current_seeds = []
    complx = None
    pocket = None
    if len(fragments) > 1:
        
        if not all([7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()] 
                    for fragment in fragments]):
        
            try:
                pocket_full_path = os.path.join(CROSSDOCKED_POCKET10_PATH, pocket_path)
                pocket = Chem.MolFromPDBFile(pocket_full_path, sanitize=False)
                if pocket.GetNumAtoms() > 20 :
                    complx = Complex(ligand, protein_path, pocket_full_path)
                    # complx._pocket = Chem.MolFromPDBFile(pocket_full_path)
                    # if all([atom.GetAtomicNum() in z_list for atom in complx.pocket.mol.GetAtoms()]):
                    pf = complx.vina_protein.pdbqt_filepath
            
                    for fragment in fragments:
                        if not any([7 in [atom.GetIsotope() for atom in fragment.mol.GetAtoms()]]):
                            current_seed = Fragment.from_fragment(fragment)
                            current_seeds.append(current_seed)
            except:
                print(f'No pdbqt file for {protein_path}')
                
    return current_seeds, complx, pocket