import logging
import MDAnalysis as mda

from rdkit.Chem import Mol
from .protein import Protein
from MDAnalysis import Universe


class Pocket():
    
    def __init__(self,
                 protein: Protein,
                 native_ligand: Mol,
                 distance_from_ligand: float = 5, # Angstrom
                 pocket_filepath: str = None,
                 ) -> None:
        """
        The protein must be only the protein atoms, with hydrogens
        We put native ligand as input because we work with existing complex
        This code would need to be adapted if working with new pocket 
        that don't have known ligand
        """
        self.protein = protein
        self.native_ligand = native_ligand
        self.distance_from_ligand = distance_from_ligand
        self.pocket_filepath = pocket_filepath
        
        self.mol = self.extract_pocket_mol(universe=self.protein.universe,
                                           pocket_filepath=pocket_filepath)
        
        
    def extract_pocket_mol(self,
                           universe: Universe,
                            ligand_resname: str = 'UNL',
                            pocket_filepath: str = None):
        ligand = mda.Universe(self.native_ligand)
        ligand.add_TopologyAttr('resname', [ligand_resname])
        
        complx = mda.Merge(universe.atoms, ligand.atoms)

        selections = ['protein',
                      f'around {self.distance_from_ligand} resname {ligand_resname}',
                      'not type H']
        selection = '(' + ') and ('.join(selections) + ')'
        atom_group: mda.AtomGroup = complx.select_atoms(selection)
        # atom_group.write('test_pocket.pdb')
        pocket_mol = None
        if len(atom_group) > 20:
            segids = {}
            for residue in atom_group.residues:
                segid = residue.segid
                resid = residue.resid
                if segid in segids:
                    segids[segid].append(resid)
                else:
                    segids[segid] = [resid]
            selections = []
            for segid, resids in segids.items():
                resids_str = ' '.join([str(resid) for resid in set(resids)])
                selections.append(f'((resid {resids_str}) and (segid {segid}))')
            pocket_selection = ' or '.join(selections)
            protein_pocket: mda.AtomGroup = universe.select_atoms(pocket_selection)
            pocket_mol = protein_pocket.atoms.convert_to("RDKIT")
            if pocket_filepath is not None:
                protein_pocket.atoms.write(pocket_filepath)
        else:
            raise Exception('Pocket quite small')
            # import pdb;pdb.set_trace()
            
        return pocket_mol