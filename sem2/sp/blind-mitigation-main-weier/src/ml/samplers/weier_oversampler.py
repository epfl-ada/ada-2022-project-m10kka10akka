import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter
import json
import markovify
import random

from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

from ml.samplers.shufflers.shuffler import Shuffler
from ml.samplers.weier_scaler import Scaler

class WeierOversampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'weier scaling oversampling'
        self._notation = 'wsos'
        
        self._rebalancing_mode = self._settings['ml']['oversampler']['rebalancing_mode'] 
        self._shuffler = Shuffler(settings)
        scaler_settings = {'mode': 'factor', 'scaling_factor': 1}
        # USAGE: change the scaling settings here
        self._scaler = Scaler(scaler_settings)

    def _oversample(self, sequences:list, labels:list, oversampler:list, demographics,  sampling_strategy:dict, only:str='none') -> Tuple[list, list, list]:
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


        #### Part where you code your data augmentation technique
        # Begin block
        # 1) the indices in indices_resampled are the indices of the data you have to augment. 
        # Here, it makes sure that all sequences are at least once in their original shape in the training set
        potential_shuffles = [idx[0] for idx in indices_resampled]
        print(potential_shuffles)
        [potential_shuffles.remove(idx) for idx in range(len(sequences))]
        assert len(potential_shuffles) == (len(indices_resampled) - len(indices))
        # 1) the indices in indices_resampled are the indices of the data you have to augment. 
        # Here, it makes sure that all sequences are at least once in their original shape in the training set
        # End block

        # 2) Objects storing the sequences which you will edit. Here, I called it shuffled because I shuffled the sequences.
        shuffled_sequences = []
        shuffled_oversampler = []
        shuffled_labels = []
        shuffled_indices = []
        # 2) 

        ### Begin EDIT BLOCK
        # 3) Actual part which you can change
        
        for idx in potential_shuffles:
            if np.random.rand() < 1 / self._settings['ml']['oversampler']['shuffler']['shuffling_coin']:
                print('shuffling')
                shuffled_sequences.append(self._shuffler.shuffle(sequences[idx]))
                #shuffled_sequences.append(self._scaler.scale(sequences[idx]))
                # USAGE: change the mode here, shuffle or scaling
                
            else:
                # print('not shuffling')
                shuffled_sequences.append(sequences[idx])
        # 3) Actual part which you can implement for your data augmentation
        ### End EDIT BLOCK

            # Saving the data
            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

        # Adding the original sequences
        [shuffled_sequences.append(sequences[idx]) for idx in range(len(sequences))]
        [shuffled_labels.append(labels[idx]) for idx in range(len(labels))]
        [shuffled_indices.append(idx) for idx in range(len(labels))]
        [shuffled_oversampler.append(oversampler[idx]) for idx in range(len(oversampler))]


        print('distrbution os after the sampling: {}'.format(sorted(Counter(shuffled_oversampler).items())))
        print('labels after sampling: {}'.format(Counter(shuffled_labels)))
        return shuffled_sequences, shuffled_labels, shuffled_indices     


    def _equal_oversampling(self, sequences:list, oversampler:list, labels:list, demographics:dict) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Rebalances all classes equally

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        return self._oversample(sequences, labels, oversampler, demographics, 'all')

    def sample(self, sequences:list, oversampler:list, labels:list, demographics:list) -> Tuple[list, list]:
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
            return self._equal_oversampling(sequences, oversampler, labels, demographics)
        
    def get_indices(self) -> np.array:
        return self._indices