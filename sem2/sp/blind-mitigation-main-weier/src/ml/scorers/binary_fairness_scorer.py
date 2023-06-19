import numpy as np

from ml.scorers.scorer import Scorer

class BinaryFairnessScorer(Scorer):
    """This class takes care of computing fairness metrics

    Args:
        Scorer (Scorer): Inherits from Scorer
    """

    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'fairness scorer'
        self._notation = 'fscr'

