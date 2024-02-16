from rdkit.Chem import BRICS
from scipy.spatial import distance_matrix
from ymir.utils.fragment import (get_neighbor_id_for_atom_id, 
                                   get_max_num_neighbor_from_original)
from collections import defaultdict
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
from ymir.data import Fragment
from rdkit.Chem.rdMolAlign import AlignMol

potential_reaction_t = (int, str)
potential_reactions: dict[int, dict[int, potential_reaction_t]] = defaultdict(dict)
reaction_defs = BRICS.reactionDefs
rxn_i = 0
for l in reaction_defs:
    for point1, point2, bond_type in l:
        point1 = int(point1.replace('a', '').replace('b', ''))
        point2 = int(point2.replace('a', '').replace('b', ''))
        t = (rxn_i, bond_type)
        potential_reactions[point1][point2] = t
        potential_reactions[point2][point1] = t
        rxn_i += 1
        
reactions = BRICS.reverseReactions

# def BRICSBuildStep(seed: Mol,
#                    fragments: Mol,
#                    seed_protections: dict[int, int],
#                    fragments_protections: dict[int, int],
#                    ):

#     reactions = BRICS.reverseReactions
          
#     for rxn in reactions:

#         seedIsR1 = False
#         seedIsR2 = False
        
#         if seed.HasSubstructMatch(rxn._matchers[0]):
#             seedIsR1=True
#         if seed.HasSubstructMatch(rxn._matchers[1]):
#             seedIsR2=True
        
#         for fragment, fragment_protections in zip(fragments, fragments_protections):
            
#             reactions_products = None
#             if fragment.HasSubstructMatch(rxn._matchers[0]):
#                 if seedIsR2:
#                     reactions_products = rxn.RunReactants((fragment,seed))
#             if fragment.HasSubstructMatch(rxn._matchers[1]):
#                 if seedIsR1:
#                     reactions_products = rxn.RunReactants((seed,fragment))
                    
#             if reactions_products:
                
#                 for products in reactions_products:
#                     product = products[0]

#                     product_positions = product.GetConformer().GetPositions()
#                     seed_positions = seed.GetConformer().GetPositions()
                    
#                     dists = distance_matrix(seed_positions, product_positions)
#                     closest_id_in_seed = dists.argmin(axis=1)
#                     seed_to_product_mapping = {seed_id: product_id for product_id, seed_id in enumerate(closest_id_in_seed)}
                    
#                     fragment_positions = seed.GetConformer().GetPositions()
                    
#                     dists = distance_matrix(fragment_positions, product_positions)
#                     closest_id_in_seed = dists.argmin(axis=1)
#                     fragment_to_product_mapping = {seed_id: product_id for product_id, seed_id in enumerate(closest_id_in_seed)}
                    
#                     product_protections = {}
#                     for seed_id, attach_point in seed_protections.items():
#                         product_id = seed_to_product_mapping[seed_id]
#                         product_protections[product_id] = attach_point
#                     for seed_id, attach_point in fragment_protections.items():
#                         product_id = fragment_to_product_mapping[seed_id]
#                         product_protections[product_id] = attach_point
                        
#                     deprotect_fragment(mol=product, protections=product_protections)
                    
#                     seed_atom_ids = [product_id 
#                                      for seed_id, product_id in seed_to_product_mapping.items()
#                                      if seed_id not in seed_protections]
                    
#                     # for coords, point in protected_atoms:
#                     #     coords = np.array(coords)
#                     #     coord_match = np.any(product_positions == coords, axis=1)
#                     #     atom_id = int(np.nonzero(coord_match)[0][0])
#                     #     atom = product.GetAtomWithIdx(atom_id)
#                     #     atom.SetAtomicNum(0)
#                     #     atom.SetIsotope(point)

#                     # seed_real_ids = [atom.GetIdx() for atom in seed.GetAtoms() if not atom.GetAtomicNum() == 0]
#                     # seed_positions = seed.GetConformer().GetPositions()[np.array(seed_real_ids)]
#                     # seed_atom_ids = []
#                     # for coords in seed_positions:
#                     #     coord_match = np.any(product_positions == coords, axis=1)
#                     #     atom_id = int(np.nonzero(coord_match)[0][0])
#                     #     seed_atom_ids.append(atom_id)

#                     # complete_seed_atom_ids = set()
#                     # for atom_id in seed_atom_ids:
#                     #     atom = product.GetAtomWithIdx(atom_id)
#                     #     neighbors = atom.GetNeighbors()
#                     #     neighbor_ids = [neighbor.GetIdx() for neighbor in neighbors]
#                     #     complete_seed_atom_ids.update(neighbor_ids)
                    
#                     # yield product, complete_seed_atom_ids
#                     yield product, seed_atom_ids
                    
                    
def add_fragment_to_seed(seed: Fragment,
                         fragment: Fragment,
                         torsion_value: float):
    
    assert torsion_value >= -180 and torsion_value <= 180, 'Torsion is in [-180, 180]'
    
    seed_attach_points = seed.get_attach_points()
    assert len(seed_attach_points) == 1, 'There must be only one attach point on seed'
    
    fragment_attach_points = fragment.get_attach_points()
    assert len(fragment_attach_points) == 1, 'There must be only one attach point on fragment'
    
    seed_attach_atom_ids = list(seed_attach_points.keys())
    fragment_attach_atom_ids = list(fragment_attach_points.keys())
    
    # Align the attachment-point/neighbor axis between the seed and the added fragment
    atom_id1 = seed_attach_atom_ids[0]
    neighbor_id1 = get_neighbor_id_for_atom_id(mol=seed, atom_id=atom_id1)
    
    atom_id2 = fragment_attach_atom_ids[0]
    neighbor_id2 = get_neighbor_id_for_atom_id(mol=fragment, atom_id=atom_id2)
    
    rmsd = AlignMol(fragment, 
                    seed, 
                    atomMap=[(atom_id2, neighbor_id1), (neighbor_id2, atom_id1)])
    
    seed_attach_label: int = list(seed_attach_points.values())[0]
    fragment_attach_label: int = list(fragment_attach_points.values())[0]
    potential_reactions_seed = potential_reactions[seed_attach_label]
    
    try:
        assert fragment_attach_label in potential_reactions_seed
    except:
        import pdb;pdb.set_trace()
    
    rxn_i, bond_type = potential_reactions_seed[fragment_attach_label]
    rxn = reactions[rxn_i]
    
    reactions_products = []
    if fragment.HasSubstructMatch(rxn._matchers[0]):
        if seed.HasSubstructMatch(rxn._matchers[1]):
            reactions_products = rxn.RunReactants((fragment, seed))
    elif fragment.HasSubstructMatch(rxn._matchers[1]):
        if seed.HasSubstructMatch(rxn._matchers[0]):
            reactions_products = rxn.RunReactants((seed, fragment))
    
    try:
        assert len(reactions_products) == 1
    except:
        import pdb;pdb.set_trace()
    products = reactions_products[0]
    assert len(products) == 1
    
    product = products[0]
    
    product_positions = product.GetConformer().GetPositions()

    # real_seed_ids = [atom.GetIdx() 
    #                     for atom in seed.GetAtoms()
    #                 if atom.GetAtomicNum() != 0]
    seed_positions = seed.GetConformer().GetPositions()
    
    dists = distance_matrix(seed_positions, product_positions)
    closest_product_id_in_seed = dists.argmin(axis=1)
    seed_to_product_mapping = {seed_id: int(product_id) 
                                for seed_id, product_id in enumerate(closest_product_id_in_seed)}

    fragment_positions = fragment.GetConformer().GetPositions()
    
    dists = distance_matrix(fragment_positions, product_positions)
    closest_product_id_in_frag = dists.argmin(axis=1)
    fragment_to_product_mapping = {frag_id: int(product_id) 
                                    for frag_id, product_id in enumerate(closest_product_id_in_frag)}

    # Get anchor and attach from seed
    for atom in seed.GetAtoms():
        if atom.GetAtomicNum() == 0:
            seed_attach_atom_id = atom.GetIdx()
            break
    prod_seed_attach_atom_id = seed_to_product_mapping[seed_attach_atom_id]
    seed_anchor_atom_id = get_neighbor_id_for_atom_id(seed, seed_attach_atom_id)
    # prod_seed_anchor_atom_id = seed_to_product_mapping[seed_anchor_atom_id]

    # Get anchor and attach from fragment
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0:
            fragment_attach_atom_id = atom.GetIdx()
            break
    prod_fragment_attach_atom_id = fragment_to_product_mapping[fragment_attach_atom_id]
    fragment_anchor_atom_id = get_neighbor_id_for_atom_id(fragment, fragment_attach_atom_id)
    # prod_fragment_anchor_atom_id = fragment_to_product_mapping[fragment_anchor_atom_id]
    
    # Seed anchor = fragment attach, and fragment anchor = seed attach
    
    max_num_seed_neigh = get_max_num_neighbor_from_original(atom_id=prod_fragment_attach_atom_id, 
                                                            mol=product, 
                                                            original_atom_id=seed_anchor_atom_id) 
    max_num_fragment_neigh = get_max_num_neighbor_from_original(atom_id=prod_seed_attach_atom_id, 
                                                                mol=product, 
                                                                original_atom_id=fragment_anchor_atom_id)
    
    # Only set torsion if there is a torsion (not counting H ?)
    if max_num_seed_neigh is not None and max_num_fragment_neigh is not None:
        begin_atom_idx = max_num_seed_neigh
        second_atom_idx = prod_fragment_attach_atom_id
        third_atom_idx = prod_seed_attach_atom_id
        end_atom_idx = max_num_fragment_neigh
        SetDihedralDeg(product.GetConformer(), 
                        begin_atom_idx,
                        second_atom_idx,
                        third_atom_idx,
                        end_atom_idx,
                        torsion_value)
        
    seed_protections = seed.protections
    fragment_protections = fragment.protections
    product_protections = get_product_protections(seed_protections,
                                                  seed_to_product_mapping,
                                                  fragment_protections,
                                                  fragment_to_product_mapping)
    product = Fragment(mol=product, protections=product_protections)
    product.unprotect()
    
    return product
    
    
def get_product_protections(seed_protections,
                            seed_to_product_mapping,
                            fragment_protections,
                            fragment_to_product_mapping):
    
    # deprotect product
    product_protections = {}
    for seed_id, attach_point in seed_protections.items():
        product_id = seed_to_product_mapping[seed_id]
        product_protections[product_id] = attach_point
    for frag_id, attach_point in fragment_protections.items():
        try:
            product_id = fragment_to_product_mapping[frag_id]
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
        product_protections[product_id] = attach_point
        
    return product_protections