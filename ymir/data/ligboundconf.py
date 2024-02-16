from ymir.params import LIGBOUNDCONF_FILEPATH
from rdkit import Chem

class LigBoundConf():
    
    def __init__(self,
                 ligand_path = LIGBOUNDCONF_FILEPATH) -> None:
        self.ligand_path = ligand_path
        self.ligands = [ligand for ligand in Chem.SDMolSupplier(LIGBOUNDCONF_FILEPATH)]
        
        self.download_cif_files()
        
    # def download_cif_files():
    #     for ligand in self.ligands:
    #         ligand_name = ligand.GetProp('_Name') # LIGANDNAME_PDBID_CHAIN_RESNUM
    #         pdb_id = ligand_name.split('_')[1]
            
            
            
    #     cmd = f'wget -O {ccp4_filepath} https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4'
    #     process = subprocess.run(cmd.split())
    #     if process.returncode == 8:
    #         if os.path.exists(ccp4_filepath):
    #             os.remove(ccp4_filepath)
    #         raise WGETException('File could not be downloaded (probably 404 not found)')