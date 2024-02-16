from rdkit import Chem
from rdkit.Chem import Mol
from ccdc.io import Molecule

def rdkit_conf_to_ccdc_mol(rdkit_mol: Mol, 
                            conf_id: int = -1) -> Molecule:
    """Create a ccdc molecule for a given conformation from a rdkit molecule
    Communication via mol block
    
    :param rdkit_mol: RDKit molecule
    :type rdkit_mol: Mol
    :param conf_id: Conformer ID in the RDKit molecule
    :type conf_id: int
    :return: CCDC molecule
    :rtype: Molecule
    
    """
    molblock = Chem.MolToMolBlock(rdkit_mol, 
                                    confId=conf_id)
    molecule: Molecule = Molecule.from_string(molblock)
    return molecule

def ccdc_mol_to_rdkit_mol(ccdc_mol: Molecule) -> Mol:
    """Transforms a ccdc molecule to an rdkit molecule

    :param ccdc_mol: CCDC molecule
    :type ccdc_mol: Molecule
    :return: RDKit molecule
    :rtype: Mol
    """
    
    # First line is necessary in case the ccdc mol is a DockedLigand
    # because it contains "fake" atoms with atomic_number lower than 1
    ccdc_mol.remove_atoms([atom 
                            for atom in ccdc_mol.atoms 
                            if atom.atomic_number < 1])
    mol2block = ccdc_mol.to_string()
    
    return Chem.MolFromMol2Block(mol2block, 
                                    removeHs=False)