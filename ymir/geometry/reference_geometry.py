import os
import numpy as np
import pickle
import logging

from abc import (abstractmethod, 
                 ABC)
from typing import (List, 
                    Dict, 
                    Any, 
                    Union)
from tqdm import tqdm
from collections import defaultdict
from ymir.params import DATA_DIRPATH
from rdkit.Chem import Mol
from ccdc.io import Molecule
from ymir.utils.mol_conversion import ccdc_mol_to_rdkit_mol
from .geometry_extractor import GeometryExtractor
from sklearn.mixture import GaussianMixture
from scipy.stats import (vonmises, 
                         norm)
from sklearn.model_selection import GridSearchCV

Mixture = GaussianMixture
MixturesDict = Dict[str, Dict[str, Union[float, Mixture]]]


class ReferenceGeometry(ABC):
    
    def __init__(self,
                 root: str = DATA_DIRPATH,
                 source_name: str = None,
                 validity_method: str = 'boundaries',
                 torsion_mixture: str = 'Gaussian',
                 get_closest_mixture: bool = False) -> None:
        self.root = root
        self.source_name = source_name
        self.validity_method = validity_method
        self.torsion_mixture = torsion_mixture
        self.get_closest_mixture = get_closest_mixture
        
        self.geometry_extractor = GeometryExtractor()
        
        self.values_filepath = os.path.join(root, f'{source_name}_geometry_values.p')
        self.ranges_filepath = os.path.join(root, f'{source_name}_geometry_ranges.p')
        self.mixtures_filepath = os.path.join(root, f'{source_name}_geometry_mixtures.p')
        
        if validity_method == 'boundaries':
            self.ranges = self.read_ranges()
        else:
            self.mixtures = self.read_mixtures()
    
    
    @abstractmethod
    def get_mol_iterator(self) -> List[Mol]:
        raise NotImplementedError()
    
    
    def read_values(self):
        if not os.path.exists(self.values_filepath):
            values = self.compute_values()
        else:
            with open(self.values_filepath, 'rb') as f:
                values = pickle.load(f)
        return values
    
    
    def read_ranges(self):
        values = self.read_values()
        if not os.path.exists(self.ranges_filepath):
            ranges = self.compute_ranges(values)
        else:
            with open(self.ranges_filepath, 'rb') as f:
                ranges = pickle.load(f)
        return ranges
    
    
    def read_mixtures(self):
        values = self.read_values()
        if not os.path.exists(self.mixtures_filepath):
            mixtures = self.compute_mixtures()
        else:
            with open(self.mixtures_filepath, 'rb') as f:
                mixtures = pickle.load(f)
        return mixtures
    
    
    def compute_values(self):
        
        logging.info(f'Compiling geometry values for {self.source_name}')
        
        mol_iterator = self.get_mol_iterator()
        
        if self.validity_method == 'mixtures': # CSD
            n = 100000
            r = list(range(len(mol_iterator)))
            random_idxs = np.random.choice(r, n)
        else:
            random_idxs = range(len(mol_iterator))
        
        all_bond_values = defaultdict(list)
        all_angle_values = defaultdict(list)
        all_torsion_values = defaultdict(list)
        
        # for mol in tqdm(mol_iterator):
        for i in tqdm(random_idxs):
            original_mol = mol_iterator[int(i)]
            if isinstance(original_mol, Molecule):
                try:
                    mol = ccdc_mol_to_rdkit_mol(original_mol)
                except Exception as e:
                    logging.warning('CCDC mol could not be converted to RDKit :' + str(e))
                    mol = None
            else:
                mol = original_mol
            
            if mol is not None:
                assert isinstance(mol, Mol)
                
                mol_bond_values = self.get_mol_bond_lengths(mol)
                for bond_tuple, bond_values in mol_bond_values.items():
                    all_bond_values[bond_tuple].extend(bond_values)
                    # if bond_tuple == (((('H', 'S', 0), '-'), (('H', 'S', 0), '-'), (('H', 'S', 0), '-')), ('C', 'SP3', 0), '-', ('N', 'SP2', 0), ((('C', 'SP2', 0), '~'), (('C', 'SP2', 0), '~'))):
                    #     for value in bond_values:
                    #         if value < 1.40:
                    #             import pdb;pdb.set_trace()
                
                mol_angle_values = self.get_mol_angle_values(mol)
                for angle_tuple, angle_values in mol_angle_values.items():
                    all_angle_values[angle_tuple].extend(angle_values)
                    
                mol_torsion_values = self.get_mol_torsion_values(mol)
                for torsion_tuple, torsion_values in mol_torsion_values.items():
                    all_torsion_values[torsion_tuple].extend(torsion_values)
                    # if torsion_tuple == (('C', 'SP2', 0), '~', (('C', '~'),), ('C', 'SP2', 0), '~', ('C', 'SP2', 0), (('C', '~'),), '~', ('C', 'SP2', 0)):
                    #     for value in torsion_values:
                    #         if value < 150 and value > 125:
                    #             import pdb;pdb.set_trace()
            
        # bond_values = self.compile_bond_lengths(mols)
        # angle_values = self.compile_angle_values(mols)
        # torsion_values = self.compile_torsion_values(mols)
        
        values = {'bond': all_bond_values,
                  'angle': all_angle_values,
                  'torsion': all_torsion_values}
        
        with open(self.values_filepath, 'wb') as f:
            pickle.dump(values, f)
            
        return values
    
    
    def compute_ranges(self,
                       values: Dict[str, Dict[str, List[float]]]
                       ) -> Dict[str, List[Any]]:
        
        logging.info(f'Computing geometry ranges for {self.source_name}')
        
        ranges = self.get_ranges_from_values(bond_values=values['bond'],
                                             angle_values=values['angle'],
                                             torsion_values=values['torsion'])
        
        with open(self.ranges_filepath, 'wb') as f:
            pickle.dump(ranges, f)
            
            
    def compute_mixtures(self,
                         n_components: int = 10,
                         n_samples: int = 3600,
                        #  n_min_values: int = 20,
                         ) -> Dict[str, MixturesDict]:
        
        logging.info(f'Computing geometry mixtures for {self.source_name}')
        
        values = self.read_values()
        
        bond_values = values['bond']
        angle_values = values['angle']
        torsion_values = values['torsion']
        
        logging.info(f'Computing bond mixtures for {self.source_name}')
        bond_mixtures = {}
        for pattern, values in tqdm(bond_values.items()):
            if len(values) > 50:
                mixture = self.get_mixture(values)
                max_likelihood = self.get_max_likelihood(mixture, values)
                bond_mixtures[pattern] = {'mixture': mixture,
                                        'max_likelihood': max_likelihood}
            
        logging.info(f'Computing angle mixtures for {self.source_name}')
        angle_mixtures = {}
        for pattern, values in tqdm(angle_values.items()):
            if len(values) > 50:
                mixture = self.get_mixture(values)
                max_likelihood = self.get_max_likelihood(mixture, values)
                angle_mixtures[pattern] = {'mixture': mixture,
                                        'max_likelihood': max_likelihood}
            
        logging.info(f'Computing torsion mixtures for {self.source_name}')
        torsion_mixtures = {}
        for pattern, values in tqdm(torsion_values.items()):
            if len(values) > 50:
                logging.info(pattern)
                    
                # Find minimum in the circular distribution
                
                # count, bins = np.histogram(values, 
                #                            bins=np.linspace(-180, 180, 181))
                # argmin = np.argmin(count)
                # bin = bins[argmin:argmin+2]
                # lowest_degree = np.mean(bin)
                
                # symmetrized_values = values + [-value for value in values]
                # positive_samples = np.linspace(0, 359, 360).reshape(-1, 1) # reshape for broadcasting
                # symmetrized_values = np.array(symmetrized_values)
                # positive_values = symmetrized_values + 180
                # distances = positive_samples - positive_values
                # mod_distances = np.mod(distances, 360)
                # min_distances = np.min(mod_distances, axis=1)
                # max_i = np.argmax(min_distances)
                # lowest_degree = max_i - 180
                
                abs_values = np.abs(values)
                samples = np.linspace(0, 180, 181).reshape(-1, 1)
                distances = np.abs(samples - abs_values)
                min_distances = np.min(distances, axis=1)
                max_i = np.argmax(min_distances)
                
                # import pdb;pdb.set_trace()
                
                # shift the circular distribution putting the lowest point at -180
                shifted_values = self.shift_torsion_values(values=values,
                                                            x=max_i)

                # shifted_abs_values = np.abs(shifted_values)
                
                # mixture = self.get_mixture(shifted_abs_values)
                # max_likelihood = self.get_max_likelihood(mixture, shifted_abs_values)
                mixture = self.get_mixture(shifted_values)
                max_likelihood = self.get_max_likelihood(mixture, shifted_values)
                torsion_mixtures[pattern] = {'mixture': mixture,
                                                'shift': max_i,
                                                'max_likelihood': max_likelihood}
            
        mixtures = {'bond': bond_mixtures,
                    'angle': angle_mixtures,
                    'torsion': torsion_mixtures}
        
        with open(self.mixtures_filepath, 'wb') as f:
            pickle.dump(mixtures, f)
            
        return mixtures
            
            
    def get_mixture(self,
                    values: list[float]):
        values = np.array(values).reshape(-1, 1)
        best_gm = GaussianMixture(1)
        best_gm.fit(values)
        min_bic = best_gm.bic(values)
        for n_components in range(2, 7):
            gm = GaussianMixture(n_components)
            gm.fit(values)
            bic = gm.bic(values)
            if bic < min_bic:
                best_gm = gm
                min_bic = bic
        return best_gm
        
        grid_search = GridSearchCV(estimator=GaussianMixture(), 
                                   param_grid=param_grid,
                                   scoring=self.get_negative_bic_score,
                                   cv=1)
        
        grid_search.fit(values)
        mixture = grid_search.best_estimator_
        return mixture
        
    # @staticmethod
    # def get_negative_bic_score(estimator, X):
    #     bic = estimator.bic(X)
    #     # GridSearchCV maximizes a value, and we want the lowest BIC, so we negate
    #     return -bic
        
        
    def get_likelihoods(self,
                       mixture: GaussianMixture,
                       values: list[float]):
        means = mixture.means_.reshape(-1)
        stds = np.sqrt(mixture.covariances_).reshape(-1)
        weights = mixture.weights_.reshape(-1)
        likelihoods = []
        for mean, std, weight in zip(means, stds, weights):
            g_likelihoods = norm.pdf(values, loc=mean, scale=std)
            max_likelihood = norm.pdf(mean, loc=mean, scale=std)
            norm_g_likelihoods = g_likelihoods / max_likelihood
            likelihoods.append(norm_g_likelihoods * weight)
            
        return np.sum(likelihoods, axis=0)
        
        
    def get_max_likelihood(self,
                           mixture: GaussianMixture,
                           values: list[float],
                           n_samples: int = 3600,
                           ) -> float:
        values = mixture.means_.reshape(-1)
        likelihoods = self.get_likelihoods(mixture, values)
        max_likelihood = np.max(likelihoods)
        return max_likelihood
        
       
    def shift_torsion_values(self,
                            values: list[float],
                            x: float):
        '''
        shift a distribution of degrees such that the current x becomes the -180.
        '''
        assert (x >= -180) and (x <= 180)
        values = np.array(values)
        positive_values = values + 180 # -180 
        positive_shift = x + 180
        shifted_values = positive_values - positive_shift
        new_values = shifted_values % 360
        centred_new_values = new_values - 180
        return centred_new_values
    
    
    def unshift_torsion_values(self,
                                values: list[float],
                                x: float):
        '''
        shift a distribution of degrees such that the current x becomes the 0.
        '''
        return self.shift_torsion_values(values, x=-x)
            
            
    def shift_abs_torsion_values(self,
                                values: list[float],
                                x: float):
        '''
        shift a distribution of degrees such that the current x becomes the 0.
        '''
        return (values - x) % 180
        
        
    def unshift_abs_torsion_values(self,
                                   values: list[float],
                                   x: float):
        '''
        reverse shift: current 0 of a degree distribution becomes the new x
        '''
        return (values + x) % 180
            
            
    def get_ranges_from_values(self,
                               bond_values: Dict[str, List[float]],
                                angle_values: Dict[str, List[float]],
                                torsion_values: Dict[str, List[float]]):
        
        bond_ranges = {}
        for bond_tuple, values in bond_values.items():
            ranges = self.compute_authorized_ranges_bond(values)
            bond_ranges[bond_tuple] = ranges
        
        angle_ranges = {}
        for angle_tuple, values in angle_values.items():
            ranges = self.compute_authorized_ranges_angle(values)
            angle_ranges[angle_tuple] = ranges
            
        torsion_ranges = {}
        for torsion_tuple, values in torsion_values.items():
            ranges = self.compute_authorized_ranges_torsion(values)
            torsion_ranges[torsion_tuple] = ranges
        
        ranges = {
            'bond': bond_ranges,
            'angle': angle_ranges,
            'torsion': torsion_ranges,
            }
        
        return ranges
              
              
    def get_mol_bond_lengths(self,
                             mol: Mol) -> Dict[str, List[float]]:
        bond_values = defaultdict(list)
        for bond in mol.GetBonds():
            bond_tuple = self.geometry_extractor.get_bond_tuple(bond)
            for conf in mol.GetConformers():
                bond_length = self.geometry_extractor.get_bond_length(conf, bond)
                bond_values[bond_tuple].append(bond_length)
        return bond_values
    
    
    def get_mol_angle_values(self,
                             mol: Mol) -> Dict[str, List[float]]:
        angle_values = defaultdict(list)
        angles_atom_ids = self.geometry_extractor.get_angles_atom_ids(mol)
        for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids :
            
            angle_tuple = self.geometry_extractor.get_angle_tuple(mol, 
                                                                    begin_atom_idx, 
                                                                    second_atom_idx, 
                                                                    end_atom_idx)
                
            for conf in mol.GetConformers():
                angle_value = self.geometry_extractor.get_angle_value(conf,
                                                                        begin_atom_idx,
                                                                        second_atom_idx,
                                                                        end_atom_idx)
                angle_values[angle_tuple].append(angle_value)
                
        return angle_values
        
                    
    def get_mol_torsion_values(self,
                               mol: Mol,
                               ) -> Dict[str, List[float]]:
        
        torsion_values = defaultdict(list)
        torsion_atom_ids = self.geometry_extractor.get_torsions_atom_ids(mol)
        for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in torsion_atom_ids :
            
            torsion_tuple = self.geometry_extractor.get_torsion_tuple(mol,
                                                                        begin_atom_idx,
                                                                        second_atom_idx,
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                
            for conf in mol.GetConformers():
                torsion_value = self.geometry_extractor.get_torsion_value(conf,
                                                                            begin_atom_idx,
                                                                            second_atom_idx,
                                                                            third_atom_idx,
                                                                            end_atom_idx)
                if not np.isnan(torsion_value):
                    torsion_values[torsion_tuple].append(torsion_value)
                        
        return torsion_values
    
    
    def compute_authorized_ranges_angle(self,
                                        values, 
                                        nbins: int = 36, 
                                        binrange: List[float] = [0.0, 180.0], 
                                        authorized_distance: float = 2.5 # Deg
                                        ) -> List[tuple]:
        if len(values) > 20:
            counts, binticks = np.histogram(values, 
                                            bins=nbins, 
                                            range=binrange)
            authorized_ranges = []
            in_range = False # cursor to indicate whether we are inside possible value range
            # We scan all bins of the histogram, and merge bins accordinly to create ranges
            for i, (value, tick) in enumerate(zip(counts, binticks)) :
                if value > 0:
                    if not in_range: # if we have values and we were not in range, we enter in range
                        start = tick
                        in_range = True
                else:
                    if in_range: # if we have no values and we were in range, we exit range
                        end = tick
                        authorized_ranges.append((start, end))
                        in_range = False
            if in_range:
                end = binticks[-1]
                authorized_ranges.append((start, end))

            new_ranges = []
            for start, end in authorized_ranges:
                new_start = max(start - authorized_distance, binrange[0])
                new_end = min(end + authorized_distance, binrange[1])
                new_range = (new_start, new_end)
                new_ranges.append(new_range)

            corrected_ranges = []
            previous_range = new_ranges[0]
            for new_range in new_ranges[1:]:
                if new_range[0] <= previous_range[1]:
                    previous_range = (previous_range[0], new_range[1])
                else:
                    corrected_ranges.append(previous_range)
                    previous_range = new_range
            corrected_ranges.append(previous_range)

            return corrected_ranges
        else:
            return [tuple(binrange)]
        
    
    def compute_authorized_ranges_bond(self,
                                       values, 
                                       authorized_distance: float = 0.025):
        min_value = np.around(np.min(values) - authorized_distance, 3)
        max_value = np.around(np.max(values) + authorized_distance, 3)
        return [(min_value, max_value)]
    
    
    def compute_authorized_ranges_torsion(self,
                                          values, 
                                        nbins: int = 36, 
                                        binrange: List[float] = [0.0, 180.0], 
                                        authorized_distance: float = 2.5,
                                        absolute_torsion: bool = True):
        if absolute_torsion:
            values = np.abs(values)
        return self.compute_authorized_ranges_angle(values, 
                                                    nbins,
                                                    binrange, 
                                                    authorized_distance)
        
    
    def geometry_is_valid(self, 
                          geometry_tuple: tuple, 
                          value: float, 
                          geometry: str = 'bond',
                          new_pattern_is_valid: bool = True,
                          ) -> tuple[float, bool]:
        
        assert geometry in ['bond', 'angle', 'torsion']
        
        if self.validity_method == 'boundaries':
        
            if geometry == 'bond':
                ranges = self.ranges['bond']
            elif geometry == 'angle' :
                ranges = self.ranges['angle']
            elif geometry == 'torsion':
                ranges = self.ranges['torsion']
                value = np.abs(value) # this method is done on absolute torsion ranges [0, 180]
            else:
                raise RuntimeError()
            
            in_range = False
            new_pattern = False
            if geometry_tuple in ranges:
                current_ranges = ranges[geometry_tuple]

                for range in current_ranges:
                    if value >= range[0] and value <= range[1]:
                        in_range = True
                        break
                    
            else:
                new_pattern = True
                if new_pattern_is_valid:
                    in_range = True
                
            q_value = float(in_range)
        
        elif self.validity_method == 'mixtures':
            
            if geometry == 'bond':
                mixtures: MixturesDict = self.mixtures['bond']
            elif geometry == 'angle' :
                mixtures: MixturesDict = self.mixtures['angle']
            elif geometry == 'torsion':
                mixtures: MixturesDict = self.mixtures['torsion']
                
            else:
                raise RuntimeError()
            
            q_value = np.nan
            new_pattern = False
            if geometry_tuple in mixtures:
                mixture_data = mixtures[geometry_tuple]
                mixture = mixture_data['mixture']
                max_likelihood = mixture_data['max_likelihood']
                if geometry == 'torsion' :
                    shift = mixture_data['shift']
                    centre_bond_order = geometry_tuple[3]
                    if centre_bond_order == '#':
                        q_value = 1 # triple bond can have very variable torsion angles
                    else:
                        if self.get_closest_mixture:
                            if self.torsion_mixture == 'VonMises':
                                value = np.deg2rad(value) # VonMises on [-pi, pi]
                                # we take the closest peak in the mixture
                                mus, kappas = mixture.means_, mixture.covariances_
                                mus = mus.reshape(-1)
                                kappas = kappas.reshape(-1)
                                distances = np.sqrt(np.square(np.cos(mus) - np.cos(value)) 
                                                - np.square(np.sin(mus) - np.sin(value)))
                                min_idx = np.argmin(distances)
                                mu = mus[min_idx]
                                kappa = kappas[min_idx]
                                likelihood = vonmises.pdf(value, loc=mu, kappa=kappa)
                                max_likelihood = vonmises.pdf(mu, loc=mu, kappa=kappa)
                                q_value = likelihood.item() / max_likelihood
                            else: # Gaussian
                                means, vars = mixture.means_, mixture.covariances_
                                means = means.reshape(-1)
                                vars = vars.reshape(-1)
                                assert means.shape == vars.shape
                                stds = np.sqrt(vars)
                                abs_value = np.abs(value)
                                
                                # shift to match mixture space, that put the minimum
                                shifted_value = self.shift_abs_torsion_values(values=abs_value, 
                                                                    x=shift)
                                distances = np.sqrt(np.square(means - shifted_value))
                                min_idx = np.argmin(distances)
                                mean = means[min_idx]
                                std = stds[min_idx]
                                likelihood = norm.pdf(shifted_value, 
                                                    loc=mean, 
                                                    scale=std)
                                max_likelihood = norm.pdf(mean, 
                                                            loc=mean, 
                                                            scale=std)
                                q_value = likelihood.item() / max_likelihood
                        else:
                            values = [value]
                            values = self.shift_torsion_values(values=values, x=shift)
                            likelihood = self.get_likelihoods(mixture, values=values)
                            q_value = likelihood.item() / max_likelihood
                
                else: # bond, angle
                    values = [value]
                    likelihood = self.get_likelihoods(mixture, values=values)
                    q_value = likelihood.item() / max_likelihood
                    
            else:
                new_pattern = True
                
            # if q_value > 1:
            #     print('q Higher than 1')
            #     import pdb;pdb.set_trace()
                
            q_value = np.min([1, q_value]) # we might not have sampled the best likelihood
                
        return q_value, new_pattern
        
        