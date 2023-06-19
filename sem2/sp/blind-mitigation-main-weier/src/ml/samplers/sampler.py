import numpy as np
import pandas as pd
from typing import Tuple

class Sampler:
    """This class is used in the cross validation part, to change the distribution of the training data
    """
    
    def __init__(self, settings):
        self._name = 'sampler'
        self._notation = 'splr'
        self._settings = settings
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
    
    def sample(self, x: list, y: list) -> Tuple[list, list]:
        """This function changes the distribution of the data passed

        Args:
            x (list): features
            y (list): labels

        Returns:
            x_resampled (list): features with the new distribution
            y_resampled (list): labels for the rebalanced features
        """
        raise NotImplementedError

    def get_indices(self) -> list:
        """Returns the indexes chosen for the resampling

        Returns:
            list: indexes from the input
        """
        raise NotImplementedError
