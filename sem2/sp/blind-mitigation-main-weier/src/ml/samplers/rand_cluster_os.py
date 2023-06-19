from importlib.metadata import distribution
import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter

from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

class RandomClusterOversampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'random oversampling'
        self._notation = 'rdmos'

    def _get_strategy_sample(self):
        if self._settings['ml']['oversampler']['strategy'] == 'maximum':
            self._sample = self._max_sample

        if self._settings['ml']['oversampler']['strategy'] == 'one':
            self._sample = self._one_sample

    def _max_sample(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        More specifically, makes sure all clusters are represented by *max_n* samples, max_n being the size of the largest cluster.
        The non-determined cluster is set to its original size (noisy=original) or to 0 (noisy=zero)

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        max_number = np.max([distribution_os[os_type] for os_type in distribution_os])
        sampler = {cluster: max_number for cluster in distribution_os}
        if '-1' in sampler:
            if self._settings['ml']['oversampler']['noisy'] == 'original':
                sampler['-1'] = distribution_os['-1']

        self._ros = ros(random_state=self._settings['seeds']['oversampler'], sampling_strategy=sampler)

        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it
        x_resampled, oversampler_resampled = self._ros.fit_resample(x, oversampler)
        sequence_resampled = [xrs[0] for xrs in x_resampled]
        y_resampled = [xrs[1] for xrs in x_resampled]
        indices_resampled = self._ros.sample_indices_

        if self._settings['ml']['oversampler']['noisy'] == 'zero' and '-1' in sampler:
            indices = [i for i in range(len(x_resampled)) if oversampler_resampled[i] != '-1']
            sequence_resampled = [sequence_resampled[idx] for idx in indices]
            oversampler_resampled = [oversampler_resampled[idx] for idx in indices]
            y_resampled = [y_resampled[idx] for idx in indices]
            indices_resampled = [indices_resampled[idx] for idx in indices]

        elif self._settings['ml']['oversampler']['noisy'] == 'highest' and '-1' in sampler:
            highest_class = max(distribution_os, key=distribution_os.get)
            indices = [i for i in range(len(x_resampled)) if oversampler_resampled[i] == highest_class]
            sequence_resampled = [sequence_resampled[idx] for idx in indices]
            oversampler_resampled = [oversampler_resampled[idx] for idx in indices]
            y_resampled = [y_resampled[idx] for idx in indices]
            indices_resampled = [indices_resampled[idx] for idx in indices]

        print('distrbution os after the sampling: {}'.format(sorted(Counter(oversampler_resampled).items())))

        print('labels before sampling: {}'.format(Counter(labels)))
        print('labels after sampling: {}'.format(Counter(y_resampled)))
        return sequence_resampled, y_resampled, indices_resampled

    def _one_sample(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        More specifically, makes sure all clusters are represented by *max_n* samples, max_n being the size of the largest cluster.
        The non-determined cluster is set to its original size (noisy=original) or to 0 (noisy=zero)

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        max_number = np.max([distribution_os[os_type] for os_type in distribution_os])
        sampler = dict(distribution_os)

        if self._settings['ml']['oversampler']['noisy'] == 'zero' and '-1' in distribution_os:
            distribution_os.pop('-1')

        if self._settings['ml']['oversampler']['one_group'] == 'largest':
            class_to_sample = max(distribution_os, key=distribution_os.get)

        if self._settings['ml']['oversampler']['one_group'] == 'smallest':
            class_to_sample = min(distribution_os, key=distribution_os.get)

        sampler[class_to_sample] = 2 * max_number

        self._ros = ros(random_state=self._settings['seeds']['oversampler'], sampling_strategy=sampler)

        x = [(sequences[idx], labels[idx]) for idx in range(len(sequences))] # ensures that we can recover the corresponding labels if our oversampler does not depend on it
        x_resampled, oversampler_resampled = self._ros.fit_resample(x, oversampler)
        sequence_resampled = [xrs[0] for xrs in x_resampled]
        y_resampled = [xrs[1] for xrs in x_resampled]
        indices_resampled = self._ros.sample_indices_

        indices = [i for i in range(len(x_resampled)) if oversampler_resampled[i] == class_to_sample]
        sequence_resampled = [sequence_resampled[idx] for idx in indices]
        oversampler_resampled = [oversampler_resampled[idx] for idx in indices]
        y_resampled = [y_resampled[idx] for idx in indices]
        indices_resampled = [indices_resampled[idx] for idx in indices]

        print('distrbution os after the sampling: {}'.format(sorted(Counter(oversampler_resampled).items())))

        print('labels before sampling: {}'.format(Counter(labels)))
        print('labels after sampling: {}'.format(Counter(y_resampled)))
        return sequence_resampled, y_resampled, indices_resampled       

    def sample(self, sequences:list, oversampler: list, labels:list) -> Tuple[list, list]:
        """
        Determines the strategy of oversampling, then samples
        """ 
        self._get_strategy_sample()
        return self._sample(sequences, oversampler, labels)

    def get_indices(self) -> np.array:
        return self._indices