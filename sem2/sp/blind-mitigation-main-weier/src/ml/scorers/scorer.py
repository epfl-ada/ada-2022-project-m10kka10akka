import numpy as np
import pandas as pd 
from typing import Tuple
from sklearn.metrics import make_scorer

class Scorer:
    """This class is the super class of all objects that score classifier performances to be passed on into the 'cross_validate' and 'gridsearch' function.
    """
    
    def __init__(self, settings):
        self._name = 'scorer'
        self._notation = 'scorer'
        self._settings = dict(settings)
        self._n_classes = self._settings['experiment']['n_classes']
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
    
    def _get_score_functions(self, settings):
        self._scorers = {}
        for score in settings['ml']['scorer']['scoring_metrics']:
            if score in self._score_dictionary:
                self._scorers[score] = self._score_dictionary[score]

    def set_optimiser_function(self, optim_scoring='roc_auc') -> make_scorer:
        """This function creates a make scorer object that calls _optim_sk_function to make the arguments (ytrue, ypred) compatible with the rest of our scorer

        Args:
            optim_scoring (str, optional): metric of the scorer. Defaults to 'roc_auc'.

        Returns:
            make_scorer: sklearn object compatible with gridsearch. Also saves it under self._optim_function
        """
        self._optim_function = self._scorers[optim_scoring]
        self._optim_scoring = optim_scoring
        self._optim_croissant = self._croissant[optim_scoring]
        
    def get_optim_function(self):
        return self._optim_function

    def get_optim_scoring(self):
        return self._optim_scoring
    
    def get_optim_croissant(self):
        return self._optim_croissant
    
    def _create_scorer_object(self):
        raise NotImplementedError
    