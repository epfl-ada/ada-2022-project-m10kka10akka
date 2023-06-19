import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter

from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

class RandomOversampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'random oversampling'
        self._notation = 'rmos'
        
        self._rebalancing_mode = self._settings['ml']['oversampler']['rebalancing_mode'] 

    def _oversample(self, x:list, oversampler:list, sampling_strategy:dict, only:str='none') -> Tuple[list, list, list]:
        """Oversamples x based on oversampler, according to the sampling_strategy.

        Args:
            x (list): concatenation of the sequences and the labels
            oversampler (list): list of the attributes by which to oversample, corresponding to the entries in x
            sampling_strategy (dict): dictionary with the keys as classes, and the values as number of samples to get, or str = 'all' if
            equally balanced
            only: if oversampling one class only, name of the class to retain
        """
        self._ros = ros(
            random_state = self._settings['seeds']['oversampler'],
            sampling_strategy=sampling_strategy
        )

        x_resampled, oversampler_resampled = self._ros.fit_resample(x, oversampler)
        sequence_resampled = [xrs[0] for xrs in x_resampled]
        y_resampled = [xrs[1] for xrs in x_resampled]
        indices_resampled = self._ros.sample_indices_


        if 'only' in self._settings['ml']['oversampler']['rebalancing_mode']:
            assert only != 'none'
            indices = [i for i in range(len(x_resampled)) if oversampler_resampled[i] == only]
            sequence_resampled = [sequence_resampled[idx] for idx in indices]
            oversampler_resampled = [oversampler_resampled[idx] for idx in indices]
            y_resampled = [y_resampled[idx] for idx in indices]
            indices_resampled = [indices_resampled[idx] for idx in indices]

        print('distrbution os after the sampling: {}'.format(sorted(Counter(oversampler_resampled).items())))
        print('labels after sampling: {}'.format(Counter(y_resampled)))
        return sequence_resampled, y_resampled, indices_resampled     


    def _equal_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Rebalances all classes equally

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it
        return self._oversample(x, oversampler, 'all')

    def _major_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples the majority class and kicks out all other instances from other classes

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        sampler = {cluster: distribution_os[cluster] for cluster in distribution_os}
        majority_class = max(distribution_os, key=distribution_os.get)
        max_number = np.max([distribution_os[os_type] for os_type in distribution_os])
        sampler[majority_class] = int(max_number * self._settings['ml']['oversampler']['oversampling_factor'])

        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        return self._oversample(x, oversampler, sampler, majority_class)   

    def _minor_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Only oversamples the minority class

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        sampler = {cluster: distribution_os[cluster] for cluster in distribution_os}

        na_keys = [] 
        if self._settings['ml']['oversampler']['na_no_minority']:
            sampler_no_na = dict(sampler)
            for key in sampler_no_na:
                if 'na' in key:
                    na_keys.append(key)
            for key in na_keys:
                del sampler[key]
            minority_class = min(sampler_no_na, key=sampler_no_na.get)
        else:
            minority_class = min(distribution_os, key=distribution_os.get)
    
        max_number = np.max([distribution_os[os_type] for os_type in distribution_os])
        sampler[minority_class] = max_number * self._settings['ml']['oversampler']['oversampling_factor']
        for key in na_keys:
            sampler[key] = distribution_os[key]
        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        return self._oversample(x, oversampler, sampler, minority_class)

    def _custom_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples the groups based on a dictionary

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        sampler = dict(self._settings['ml']['oversamper']['oversampling_distributions'])

        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        return self._oversample(x, oversampler, sampler)  

    def _cascade_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples instances from a group x up to the number of instances of the smallest-larger-than-x-group

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        sampler = {cluster: distribution_os[cluster] for cluster in distribution_os}
        na_keys = [] 
        if self._settings['ml']['oversampler']['na_no_minority']:
            for key in sampler:
                if 'na' in key:
                    na_keys.append(key)

        keys = [k for k in sampler.keys() if k not in na_keys]
        values = [sampler[k] for k in keys]
        indices = np.argsort(values)

        clusters = [keys[idx] for idx in indices]
        n_samples = [values[idx]for idx in indices]
        n_samples = n_samples[1:] + [n_samples[-1]]

        sampler = {clusters[idx]: n_samples[idx] for idx in range(len(keys))}
        for k in na_keys:
            sampler[k] = distribution_os[k]

        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        return self._oversample(x, oversampler, sampler)

    def _withingroup_balancing_oversampling(self, sequences:list, oversampler:list, labels:list, demographics:dict) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples the classes from a group up.

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """

        final_counter = {}
        groups = [ds for ds in demographics[self._settings['ml']['oversampler']['within_group']]]
        unique_groups = np.unique(groups)
        for group in unique_groups:
            group_indices = [i for i in range(len(groups)) if groups[i] == group]
            group_oversamplers = [oversampler[idx] for idx in group_indices]
            group_counter = Counter(group_oversamplers)

            group_max_number = np.max([group_counter[g] for g in group_counter])
            
            for g in group_counter:
                assert g not in final_counter
                final_counter[g] = group_max_number

        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        print('distribution of oversampled: ', final_counter)
        return self._oversample(x, oversampler, final_counter)

    def _cascade_withingroup_balancing_oversampling(self, sequences:list, oversampler:list, labels:list, demographics:dict) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples the classes from a group up.

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        final_counter = {}
        groups = [ds for ds in demographics[self._settings['ml']['oversampler']['within_group']]]
        unique_groups = np.unique(groups)
        n_groups = {}
        for group in unique_groups:
            group_indices = [i for i in range(len(groups)) if groups[i] == group]
            group_oversamplers = [oversampler[idx] for idx in group_indices]
            group_counter = Counter(group_oversamplers)
            group_max_number = np.max([group_counter[g] for g in group_counter])

            n_groups[group] = {
                'oversamplers': [go for go in group_oversamplers],
                'n': group_max_number
            }

        group_names = list(n_groups.keys())
        group_values = [n_groups[gn]['n'] for gn in group_names]
        indices = np.argsort(group_values)
        increasing_groups = [group_names[idx] for idx in indices]
        increasing_samples = [group_values[idx] for idx in indices]
        increasing_samples = increasing_samples[1:] + [increasing_samples[-1]]

        final_counter = {}
        for g_i, group in enumerate(increasing_groups):
            unique_groups_oversamplers = np.unique(n_groups[group]['oversamplers'])
            for ugo in unique_groups_oversamplers:
                final_counter[ugo] = increasing_samples[g_i]

        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        print('distribution of oversampled: ', final_counter)
        return self._oversample(x, oversampler, final_counter)

    def _majority_withingroup_balancing_oversampling(self, sequences:list, oversampler:list, labels:list, demographics:dict) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples the classes from a group up.

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        final_counter = {}
        groups = [ds for ds in demographics[self._settings['ml']['oversampler']['within_group']]]
        unique_groups = np.unique(groups)
        n_groups = {}
        for group in unique_groups:
            group_indices = [i for i in range(len(groups)) if groups[i] == group]
            group_oversamplers = [oversampler[idx] for idx in group_indices]
            group_counter = Counter(group_oversamplers)
            group_max_number = np.max([group_counter[g] for g in group_counter])

            n_groups[group] = {
                'oversamplers': [go for go in group_oversamplers],
                'n': group_max_number
            }

        group_names = list(n_groups.keys())
        group_values = [n_groups[gn]['n'] for gn in group_names]
        indices = np.argsort(group_values)
        increasing_groups = [group_names[idx] for idx in indices]
        increasing_samples = [group_values[idx] for idx in indices]
        increasing_samples = increasing_samples[:-1] + [int(increasing_samples[-1] * self._settings['ml']['oversampler']['oversampling_factor'])]

        for i in range(len(increasing_samples)):
            print(group_names[i], increasing_samples[i])
        final_counter = {}
        for g_i, group in enumerate(increasing_groups):
            unique_groups_oversamplers = np.unique(n_groups[group]['oversamplers'])
            for ugo in unique_groups_oversamplers:
                final_counter[ugo] = increasing_samples[g_i]

        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        print('distribution of oversampled: ', final_counter)
        return self._oversample(x, oversampler, final_counter)

    def _minority_withingroup_balancing_oversampling(self, sequences:list, oversampler:list, labels:list, demographics:dict) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples the classes from a group up.

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        final_counter = {}
        groups = [ds for ds in demographics[self._settings['ml']['oversampler']['within_group']]]
        unique_groups = np.unique(groups)
        n_groups = {}
        for group in unique_groups:
            group_indices = [i for i in range(len(groups)) if groups[i] == group]
            group_oversamplers = [oversampler[idx] for idx in group_indices]
            group_counter = Counter(group_oversamplers)
            group_max_number = np.max([group_counter[g] for g in group_counter])

            n_groups[group] = {
                'oversamplers': [go for go in group_oversamplers],
                'n': group_max_number
            }

        group_names = list(n_groups.keys())
        group_values = [n_groups[gn]['n'] for gn in group_names]
        indices = np.argsort(group_values)
        increasing_groups = [group_names[idx] for idx in indices]
        increasing_samples = [group_values[idx] for idx in indices]
        increasing_samples = [int(increasing_samples[-1] * self._settings['ml']['oversampler']['oversampling_factor'])] + increasing_samples[1:]

        for i in range(len(increasing_samples)):
            print(group_names[i], increasing_samples[i])
        final_counter = {}
        for g_i, group in enumerate(increasing_groups):
            unique_groups_oversamplers = np.unique(n_groups[group]['oversamplers'])
            for ugo in unique_groups_oversamplers:
                final_counter[ugo] = increasing_samples[g_i]

        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it - note to myself: could have used list of indices
        print('distribution of oversampled: ', final_counter)
        return self._oversample(x, oversampler, final_counter)

    def sample(self, sequences:list, oversampler:list, labels:list, demographics:dict) -> Tuple[list, list]:
        """Chooses the mode of oversampling

        1. equal oversampling: All instances are oversampled by n, determined by imbalanced-learn
        2. Major oversampling: Only the largest class is oversampled
        3. Only Major Oversampling: Only the largest class is oversampled, all other classes are taken out the training set
        4. Minor oversampling: Only the smallest class is oversampled
        5. Only Minor Oversampling: Only the smallest class is oversampled, all other classes are taken out the training set

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        if self._settings['ml']['oversampler']['rebalancing_mode'] == 'equal_balancing':
            return self._equal_oversampling(sequences, oversampler, labels)

        elif 'majorwithin_group' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._majority_withingroup_balancing_oversampling(sequences, oversampler, labels, demographics)

        elif 'major' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._major_oversampling(sequences, oversampler, labels)
        
        elif 'minorwithin_group' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._minority_withingroup_balancing_oversampling(sequences, oversampler, labels, demographics)

        elif 'minor' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._minor_oversampling(sequences, oversampler, labels)

        elif 'cascadewithin_group' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._cascade_withingroup_balancing_oversampling(sequences, oversampler, labels, demographics)

        elif 'cascade' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._cascade_oversampling(sequences, oversampler, labels)

        elif 'within_group' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._withingroup_balancing_oversampling(sequences, oversampler, labels, demographics)

        



    def get_indices(self) -> np.array:
        return self._indices