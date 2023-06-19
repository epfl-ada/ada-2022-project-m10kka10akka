import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import random

class Scaler:
    def __init__(self, settings: dict):
        self._name = 'scaler'
        self._settings = dict(settings)
        self._scaling_factor = self._settings.get('scaling_factor', 1)
        self._minmax_scaler = MinMaxScaler(feature_range=self._settings.get('range', (0, 1)))

    def _select_scaling(self):
        if self._settings['mode'] == 'minmax':
            return self._minmax_scale
        elif self._settings['mode'] == 'factor':
            return self._factor_scale
        else:
            raise ValueError(f"Invalid scaling mode: {self._settings['mode']}")
     
    def _minmax_scale(self, sequence: np.ndarray) -> np.ndarray:
        """ Applies min-max scaling to the input sequence, scaling only columns from the fifth column onwards.
        
        Args:
            sequence (np.ndarray): the sequence to be scaled.

        Returns:
            np.ndarray: the scaled sequence.
        """
        scaled_sequence = sequence.copy()
        scaled_sequence = np.array(scaled_sequence)
        scaled_sequence[:, 4:] = self._minmax_scaler.fit_transform(scaled_sequence[:, 4:]) * self._scaling_factor
        return scaled_sequence

    def _factor_scale(self, sequence: np.ndarray) -> np.ndarray:
        """ Scales the input sequence by multiplying only time columns by the scaling factor.

        Args:
            sequence (np.ndarray): the sequence to be scaled.

        Returns:
            np.ndarray: the scaled sequence.
        """
        scaled_sequence = sequence.copy()
        scaled_sequence = np.array(scaled_sequence)
        scaled_sequence[:, 4:] = scaled_sequence[:, 4:] * self._scaling_factor
        return scaled_sequence

    def scale(self, sequence: np.ndarray) -> np.ndarray:
        return self._select_scaling()(sequence)

    def fit_transform(self, sequence: np.ndarray) -> np.ndarray:
        return self._select_scaling()(sequence)

