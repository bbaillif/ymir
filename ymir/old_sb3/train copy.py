import os
import copy

from tqdm import tqdm
from rdkit import Chem
from ymir.data.pdbbind import PDBbind
from ymir.data.fragment import (get_unique_fragments_from_mols,
                             protect_fragment,
                             deprotect_fragment,
                             center_fragment,
                             get_attach_points)
from ymir.env import FragmentBuilderEnv
from ymir.policy import FragmentBuilderPolicy
from ymir.params import TORSION_SPACE_STEP
from ymir.molecule_builder import potential_reactions
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.Descriptors import MolWt
from ccdc.conformer import ConformerGenerator
from ymir.utils.mol_conversion import rdkit_conf_to_ccdc_mol, ccdc_mol_to_rdkit_mol
from collections import namedtuple
from ymir.data.structure import Complex
from torch.utils.tensorboard import SummaryWriter

pdbbind = PDBbind()

pdbbind_ligand_path = '/home/bb596/hdd/ymir/pdbbind_ligands.sdf'
if not os.path.exists(pdbbind_ligand_path):
    ligands = pdbbind.get_ligands()
    with Chem.SDWriter('/home/bb596/hdd/ymir/pdbbind_ligands.sdf') as writer:
        for mol in ligands:
            pdb_id = mol.GetConformer().GetProp('PDB_ID')
            mol.SetProp('PDB_ID', pdb_id)
            writer.write(mol)

ligands = [mol for mol in Chem.SDMolSupplier(pdbbind_ligand_path)]

# Load fragments
fragments_path = '/home/bb596/hdd/ymir/small_fragments_3D.sdf' 
if not os.path.exists(fragments_path):
    all_fragments = get_unique_fragments_from_mols(mols=ligands)
    small_fragments = [frag for frag in all_fragments 
                        if (frag.GetNumAtoms() <=25) 
                        and (CalcNumRotatableBonds(frag) <= 3) 
                        and (MolWt(frag) <= 150)]

    cg = ConformerGenerator()

    small_frags_copy = copy.deepcopy(small_fragments)
    small_frags_3D = []
    for frag in tqdm(small_frags_copy):

        protections = protect_fragment(frag)

        ccdc_mol = rdkit_conf_to_ccdc_mol(frag)
        conformer_hits = cg.generate(ccdc_mol)
        for conformer_hit in conformer_hits:
            conformer = conformer_hit.molecule
            rdkit_conf = ccdc_mol_to_rdkit_mol(conformer)
            deprotect_fragment(rdkit_conf, protections)
            small_frags_3D.append(rdkit_conf)

    for frag in small_frags_3D:
        center_fragment(frag)
        
    with Chem.SDWriter(fragments_path) as writer:
        for mol in small_frags_3D:
            writer.write(mol)
            
else:
    small_frags_3D = [mol for mol in Chem.SDMolSupplier(fragments_path)]
    
fragments = []
fragments_protections = []
for frag in small_frags_3D:
    attach_point = get_attach_points(mol=frag)
    if not 7 in list(attach_point.values()): # 7 = 7 reaction does not work for some reason...
        for atom_id, label in attach_point.items():
            fragment = copy.deepcopy(frag)
            fragment_protections = protect_fragment(mol=fragment,
                                                    atom_ids_to_keep=[atom_id])
            fragments.append(fragment)
            fragments_protections.append(fragment_protections)
        
print(f'There are {len(fragments)} attachable fragments in the dataset')

protein_paths = []
for ligand in ligands:
    pdb_id = ligand.GetProp('PDB_ID')
    protein_path, _ = pdbbind.get_pdb_id_pathes(pdb_id)
    protein_paths.append(protein_path)
    
assert len(ligands) == len(protein_paths)

not_working_protein = []
complexes = []
for protein_path, ligand in zip(protein_paths[:10], ligands[:10]):
    try:
        complx = Complex(ligand, protein_path) 
    except Exception as e:
        print(e)
    else:
        complexes.append(complx)

torsion_values = [v - 180 for v in range(0, 360, TORSION_SPACE_STEP)]
actions = []
Action = namedtuple("Action", "frag_i attach_label torsion_value")
set_attach_labels = set()
for frag_i, frag in enumerate(fragments):
    attach_points = get_attach_points(frag)
    for attach_atom_id, attach_label in attach_points.items():
        for torsion_value in torsion_values:
            set_attach_labels.add(attach_label)
            action = Action(frag_i, attach_label, torsion_value)
            actions.append(action)
action_dim = len(actions)

valid_action_masks = {}
for attach_label_1, d_potential_attach in potential_reactions.items():
    mask = [0 for _ in range(action_dim)]
    for act_i, action in enumerate(actions):
        if action.attach_label in d_potential_attach:
            mask[act_i] = 1
    valid_action_masks[attach_label_1] = mask

node_dim = 4 # positions (3) + atomic num
torsion_space_step = TORSION_SPACE_STEP

env = FragmentBuilderEnv(fragments=fragments,
                         fragments_protections=fragments_protections,
                         complexes=complexes,
                         valid_action_masks=valid_action_masks,
                         action_dim=action_dim,
                         node_dim=node_dim,
                         torsion_space_step=torsion_space_step)

def mask_fn(env: FragmentBuilderEnv) -> list[bool]:
    return env.get_valid_action_mask()
env = ActionMasker(env, mask_fn)

policy_class = FragmentBuilderPolicy
model = MaskablePPO(policy=policy_class,
                    env=env,
                    verbose=1,
                    tensorboard_log='logs/',
                    batch_size=4)

model.learn(1000)

model.save('checkpoint.pt')