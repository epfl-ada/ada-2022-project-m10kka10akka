import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter

from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

from ml.samplers.shufflers.shuffler import Shuffler

class TimeSeriesShufflingOversampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'time series-shuffling oversampling'
        self._notation = 'tssos'
        
        self._rebalancing_mode = self._settings['ml']['oversampler']['rebalancing_mode'] 
        self._shuffler = Shuffler(settings)

    def _oversample(self, sequences:list, labels:list, oversampler:list, sampling_strategy:dict, only:str='none') -> Tuple[list, list, list]:
        """Oversamples x based on oversampler, according to the sampling_strategy.

        Args:
            sequences (list): sequences of interaction
            labels (list): target
            oversampler (list): list of the attributes by which to oversample, corresponding to the entries in x
            sampling_strategy (dict): dictionary with the keys as classes, and the values as number of samples to get, or str = 'all' if
            equally balanced
            only: if oversampling one class only, name of the class to retain
        """
        assert len(labels) == len(sequences)
        self._ros = ros(
            random_state = self._settings['seeds']['oversampler'],
            sampling_strategy=sampling_strategy
        )

        indices = [[idx] for idx in range(len(sequences))]
        indices_resampled, _ = self._ros.fit_resample(indices, oversampler)

        potential_shuffles = [idx[0] for idx in indices_resampled]
        print(potential_shuffles)
        [potential_shuffles.remove(idx) for idx in range(len(sequences))]
        assert len(potential_shuffles) == (len(indices_resampled) - len(indices))

        shuffled_sequences = []
        shuffled_oversampler = []
        shuffled_labels = []
        shuffled_indices = []
        for idx in potential_shuffles:
            if np.random.rand() < 1 / self._settings['ml']['oversampler']['shuffler']['shuffling_coin']:
                # print('shuffling')
                shuffled_sequences.append(self._shuffler.shuffle(sequences[idx]))
            else:
                # print('not shuffling')
                shuffled_sequences.append(sequences[idx])

            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

        [shuffled_sequences.append(sequences[idx]) for idx in range(len(sequences))]
        [shuffled_labels.append(labels[idx]) for idx in range(len(labels))]
        [shuffled_indices.append(idx) for idx in range(len(labels))]
        [shuffled_oversampler.append(oversampler[idx]) for idx in range(len(oversampler))]


        if 'only' in self._settings['ml']['oversampler']['rebalancing_mode']:
            assert only != 'none'
            indices = [i for i in range(len(shuffled_oversampler)) if (shuffled_oversampler[i]) == (only)]
            shuffled_sequences = [shuffled_sequences[idx] for idx in indices]
            shuffled_oversampler = [shuffled_oversampler[idx] for idx in indices]
            shuffled_labels = [shuffled_labels[idx] for idx in indices]
            shuffled_indices = [shuffled_indices[idx] for idx in indices]

        print('distrbution os after the sampling: {}'.format(sorted(Counter(shuffled_oversampler).items())))
        print('labels after sampling: {}'.format(Counter(shuffled_labels)))
        return shuffled_sequences, shuffled_labels, shuffled_indices     


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
        return self._oversample(sequences, labels, oversampler, 'all')

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
        sampler[majority_class] = max_number * self._settings['ml']['oversampler']['oversampling_factor']

        return self._oversample(sequences, labels, oversampler, sampler, majority_class)   

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

        minority_class = min(distribution_os, key=distribution_os.get)
        max_number = np.max([distribution_os[os_type] for os_type in distribution_os])
        sampler[minority_class] = max_number * self._settings['ml']['oversampler']['oversampling_factor']
        self._ros = ros(random_state=self._settings['seeds']['oversampler'], sampling_strategy=sampler)

        return self._oversample(sequences, labels, oversampler, sampler, minority_class)

    def sample(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
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

        elif 'major' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._major_oversampling(sequences, oversampler, labels)
        
        elif 'minor' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._minor_oversampling(sequences, oversampler, labels)


    def get_indices(self) -> np.array:
        return self._indices