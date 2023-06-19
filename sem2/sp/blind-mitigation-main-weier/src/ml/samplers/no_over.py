import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter

from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

class NoSampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'random oversampling'
        self._notation = 'rdmos'
        
        self._rebalancing_mode = self._settings['ml']['oversampler']['rebalancing_mode'] 
        self._ros = ros(random_state=settings['seeds']['oversampler'], sampling_strategy=self._rebalancing_mode)
        
    def sample(self, sequences:list, oversampler:list, labels:list, demographics:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        return sequences, labels, list(range(len(sequences)))        

    def get_indices(self) -> np.array:
        return self._indices