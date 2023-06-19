import logging
import numpy as np
import pandas as pd
from typing import Tuple

from ml.scorers.scorer import Scorer
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy as CCE
from sklearn.metrics import roc_auc_score

class MultiClfScorer(Scorer):
    """This class is used to create a scorer object tailored towards multi class classification problems

    Args:
        Scorer ([type]): inherits from scorer
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'multi class classification scorer'
        self._notation = 'multiclfscorer'
        self._score_dictionary = {
            'cce' : self._get_cce,
            'accuracy': self._get_accuracy,
            'balanced_accuracy': self._get_balanced_accuracy,
            'balanced_auc': self._get_balanced_auc,
            'overall_auc': self._get_overall_auc,
            'roc': self._get_balanced_auc
        }
        self._croissant = {
            'cce': False,
            'accuracy': True,
            'balanced_accuracy': True,
            'balanced_auc': True,
            'overall_auc': True,
            'roc': True
        }
        
        self._get_score_functions(settings)
        self._cce = CCE()

    def _onehot(self, index):
        vec = list(np.zeros(self._n_classes))
        vec[index] = 1
        return vec
        
    def _get_cce(self, y_true:list, y_pred:list, y_probs:list) -> float:
        if len(y_true) == 0:
            return 0
        else:
            yt = [self._onehot(yy) for yy in y_true]
            return float(self._cce(yt, y_probs))
        
    def _get_accuracy(self, y_true:list, y_pred:list, y_probs:list) -> float:
        return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    
    def _get_balanced_accuracy(self, y_true:list, y_pred:list, y_probs:list) -> float:
        classes = np.unique(y_true)
        bacc = 0
        for cl in classes:
            indices = [x for x in list(range(len(y_true))) if y_true[x] == cl]
            preds = [y_pred[x] for x in indices]
            truths = [y_true[x] for x in indices]
            bacc += np.sum(np.array(preds) == np.array(truths)) / len(truths)
        bacc /= self._settings['experiment']['n_classes']
        return bacc
    
    def _get_balanced_auc(self, y_true:list, y_pred:list, y_probs:list) -> float:
        print('unique', np.unique(y_true))
        print('nclasses', self._n_classes)
        print('label', y_pred)
        if len(np.unique(y_true)) < self._n_classes:
            return -1
        # print('ytrue', y_true)
        # print('yprobs', y_probs)
        return roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    
    def _get_overall_auc(self, y_true:list, y_pred:list, y_probs:list) -> float:
        if len(np.unique(y_true)) < self._n_classes:
            return -1
        return roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        
    def get_scores(self, y_true: list, y_pred: list, y_probs: list) -> dict:
        # yt = [self.__onehot(xx) for xx in y_true]
        scores = {}
        for score in self._scorers:
            scores[score] = self._scorers[score](y_true, y_pred, y_probs)
            
        return scores
            

    # Fairness Scores
    """Parts of the code dedicated to fairness metric measures
    demographics is the list of the corresponding demographics with regards to y_true
    """
    def _true_positive(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            try:
                indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                demo_true = [y_true[idx] for idx in indices]
                demo_pred = [y_pred[idx] for idx in indices]

                positive = [i for i in range(len(demo_true)) if demo_true[i] >= self._settings['ml']['scorer']['threshold_positive_class']]
                yt = np.array([demo_true[i] for i in positive])
                yp = np.array([demo_pred[i] for i in positive])
                s = sum(yt == yp) / len(positive)
                scores[demo] = s
            except ZeroDivisionError:
                scores[demo] = -1
        return scores

    def _false_positive(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            try:
                indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                demo_true = [y_true[idx] for idx in indices]
                demo_pred = [y_pred[idx] for idx in indices]

                negatives = [i for i in range(len(demo_true)) if demo_true[i] < self._settings['ml']['scorer']['threshold_positive_class']]
                yt = np.array([demo_true[i] for i in negatives])
                yp = np.array([demo_pred[i] for i in negatives])
                s = sum(yt != yp) / len(negatives)
                scores[demo] = s
            except ZeroDivisionError:
                scores[demo] = -1
        return scores

    def _positive_pred(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            indices = [i for i in range(len(demographics)) if demographics[i] == demo]
            demo_pred = [y_pred[idx] for idx in indices]
            positive = [yy for yy in demo_pred if yy >= self._settings['ml']['scorer']['threshold_positive_class']]
            s = len(positive) / len(indices)
            scores[demo] = s
        return scores

    def _split_scores(self, y_true: list, y_pred: list, y_probs: list, demographics:list, metrics: list) -> dict:
        demos = np.unique(demographics)
        scores = {}
        for score in metrics:
            scores[score] = {}
            if score in self._score_dictionary:
                scores[score] = {}
                for demo in demos:
                    indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                    demo_true = [y_true[idx] for idx in indices]
                    demo_pred = [y_pred[idx] for idx in indices]
                    demo_probs = [y_probs[idx] for idx in indices]
                    scores[score][demo] = self._score_dictionary[score](demo_true, demo_pred, demo_probs)
        return scores

    def get_fairness_scores(self, y_true:list, y_pred:list, y_probs:list, demographics:list, metric_list:list) -> dict:
        """Returns dictionary with as first level keys the metrics, and as second level keys the
        demographics.

        Args:
            y_true (list): real labels
            y_pred (list): predicted labels (binary)
            y_probs (list): predicted labels (probability)
            demographics (list): corresponding demographics
            metrics (list): metrics to compute the scores for

        Returns:
            results (dict):
                score: 
                    demo0: value
                    ...
                    demon: value
        """
        
        metrics = [x for x in metric_list]
        scores = {}
        if 'tp' in metrics:
            scores['tp'] = self._true_positive(y_true, y_pred, y_probs, demographics)
            metrics.remove('tp')
        if 'fp' in metrics:
            scores['fp'] = self._false_positive(y_true, y_pred, y_probs, demographics)
            metrics.remove('fp')
        if 'pp' in metrics:
            scores['pp'] = self._positive_pred(y_true, y_pred, y_probs, demographics)
            metrics.remove('pp')

        s = self._split_scores(y_true, y_pred, y_probs, demographics, metrics)
        scores.update(s)
        return scores
                