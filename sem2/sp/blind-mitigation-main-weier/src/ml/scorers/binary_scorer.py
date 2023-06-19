from re import A
from sys import excepthook
import numpy as np
import pandas as pd
import logging
from typing import Tuple
from collections import Counter

from ml.scorers.scorer import Scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

class BinaryClfScorer(Scorer):
    """This class is used to create a scorer object tailored towards binary classification

    Args:
        Scorer (Scorer): Inherits from scorer
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'binary classification scorer'
        self._notation = '2clfscorer'
        self._score_dictionary = {
            'accuracy': self._get_accuracy,
            'balanced_accuracy': self._get_balanced_accuracy,
            'precision': self._get_precision,
            'recall': self._get_recall,
            'roc': self._get_roc,
            'tp': self._compute_single_tp,
            'fp': self._compute_single_fp,
            'fn': self._compute_single_fn
        }
        
        self._croissant = {
            'accuracy': True,
            'balanced_accuracy': True,
            'precision': True,
            'recall': True,
            'roc': True,
            'tp': True,
            'fp': False,
            'fn': False
        }

        self._get_score_functions(settings)
        

    # Performance Scores
    def _get_accuracy(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return accuracy_score(y_true, y_pred)
    
    def _get_balanced_accuracy(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return balanced_accuracy_score(y_true, y_pred)
    
    def _get_precision(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return precision_score(y_true, y_pred)
    
    def _get_recall(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return recall_score(y_true, y_pred)
    
    def _get_roc(self, y_true: list, y_pred: list, y_probs: list) -> float:
        if len(np.unique(y_true)) == 1:
            return -1
        return roc_auc_score(y_true, np.array(y_probs)[:, 1])
    
    def get_scores(self, y_true: list, y_pred: list, y_probs: list) -> dict:
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

                positive = [i for i in range(len(demo_true)) if demo_true[i] == 1]
                yt = np.array([demo_true[i] for i in positive])
                yp = np.array([demo_pred[i] for i in positive])
                s = sum(yt == yp) / len(positive)
                scores[demo] = s
            except ZeroDivisionError:
                scores[demo] = -1
        return scores

    def _compute_single_tp(self, y_true:list, y_pred:list, y_probs:list) -> float:
        try:
            positive = [i for i in range(len(y_true)) if y_true[i] == 1]
            yt = np.array([y_true[i] for i in positive])
            yp = np.array([y_pred[i] for i in positive])
            s = sum(yt == yp) / len(positive)
        except ZeroDivisionError:
            s = -1
        return s

    def _false_positive(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            try:
                indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                demo_true = [y_true[idx] for idx in indices]
                demo_pred = [y_pred[idx] for idx in indices]

                negatives = [i for i in range(len(demo_true)) if demo_true[i] == 0]
                yt = np.array([demo_true[i] for i in negatives])
                yp = np.array([demo_pred[i] for i in negatives])
                s = sum(yt != yp) / len(negatives)
                scores[demo] = s
            except ZeroDivisionError:
                scores[demo] = -1
        return scores

    def _compute_single_fp(self, y_true:list, y_pred:list, y_probs:list) -> float:
        try:
            negatives = [i for i in range(len(y_true)) if y_true[i] == 0]
            yf = np.array([y_true[i] for i in negatives])
            yp = np.array([y_pred[i] for i in negatives])
            s = sum(yf != yp) / len(negatives)
        except ZeroDivisionError:
            s = -1
        return s

    def _positive_pred(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            indices = [i for i in range(len(demographics)) if demographics[i] == demo]
            demo_pred = [y_pred[idx] for idx in indices]
            positive = [yy for yy in demo_pred if yy == 1]
            s = len(positive) / len(indices)
            scores[demo] = s
        return scores

    def _compute_single_positive_pred(self, y_true:list, y_pred:list, y_probs:list):
        try:
            pred_pos = [i for i in range(len(y_pred)) if y_pred[i] == 1]
            s = len(pred_pos) / len(y_pred)
        except ZeroDivisionError:
            s = -1

        return s

    def _false_negative(self, y_true: list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            try:
                indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                demo_true = [y_true[idx] for idx in indices]
                demo_pred = [y_pred[idx] for idx in indices]

                pos_idx = [i for i in range(len(demo_true)) if demo_true[i] == 1]
                ps = len(pos_idx)
                tps = len([demo_pred[idx] for idx in pos_idx if demo_pred[idx] == 1])
                fns = ps - tps

                scores[demo] = fns / (fns + tps)
            except ZeroDivisionError:
                scores[demo] = -1
                continue
        return scores

    def _compute_single_fn(self, y_true:list, y_pred:list, y_probs:list) -> float: 
        try:
            pos_idx = [i for i in range(len(y_true)) if y_true[i] == 1]
            ps = len(pos_idx)
            tps = len([y_pred[idx] for idx in pos_idx if y_pred[idx] == 1])
            fns = ps - tps
            s = fns / (fns + tps)
        except ZeroDivisionError:
            s = -1
        return s

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
        if 'fn' in metrics:
            scores['fn'] = self._false_negative(y_true, y_pred, y_probs, demographics)

        s = self._split_scores(y_true, y_pred, y_probs, demographics, metrics)
        scores.update(s)
        return scores



        
        