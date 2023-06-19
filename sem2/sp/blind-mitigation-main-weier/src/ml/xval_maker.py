import yaml
import logging
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model
from ml.models.ts_attention import TimestepAttentionModel
from ml.models.flipped_bilstm import FlippedBiLSTMModel
from ml.models.tuglet_lstm import TugletLSTMModel

from ml.samplers.sampler import Sampler
from ml.samplers.rand_over_sampler import RandomOversampler
from ml.samplers.rand_cluster_os import RandomClusterOversampler
from ml.samplers.ts_shuffling_oversampler import TimeSeriesShufflingOversampler
from ml.samplers.no_over import NoSampler
from ml.samplers.weier_oversampler import WeierOversampler
from ml.samplers.markov_oversampler import MarkovOversampler

from ml.scorers.scorer import Scorer
from ml.scorers.binary_scorer import BinaryClfScorer
from ml.scorers.multi_scorer import MultiClfScorer

from ml.splitters.splitter import Splitter
from ml.splitters.stratified_split import MultipleStratifiedKSplit

from ml.crossvalidators.crossvalidator import CrossValidator
from ml.crossvalidators.non_nested_cv import NonNestedRankingCrossVal

from ml.splitters.stratified_split import MultipleStratifiedKSplit 

class XValMaker:
    """This script assembles the machine learning component and creates the training pipeline according to:
    
        - splitter
        - sampler
        - model
        - xvalidator
        - scorer
    """
    
    def __init__(self, settings:dict):
        logging.debug('initialising the xval')
        self._name = 'training maker'
        self._notation = 'trnmkr'
        self._settings = dict(settings)
        self._experiment_root = self._settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._pipeline_settings = self._settings['ml']['pipeline']
        
        self._build_pipeline()
        
    # def _choose_splitter(self):
    #     if self._pipeline_settings['splitter'] == 'stratkf':
    #         self._splitter = StratifiedKSplit

    def get_gridsearch_splitter(self):
        return self._gs_splitter

    def get_sampler(self):
        return self._sampler

    def get_scorer(self):
        return self._scorer

    def get_model(self):
        return self._model

    def _choose_splitter(self, splitter:str) -> Splitter:
        if splitter == 'stratkf':
            self._splitter = MultipleStratifiedKSplit
        if splitter == 'none':
            self._splitter = 'none'
        return self._splitter
    
    def _choose_inner_splitter(self):
        self._inner_splitter = self._choose_splitter(self._pipeline_settings['inner_splitter'])

    def _choose_outer_splitter(self):
        self._outer_splitter = self._choose_splitter(self._pipeline_settings['outer_splitter'])

    def _choose_gridsearch_splitter(self):
        self._gs_splitter = self._choose_splitter(self._pipeline_settings['gs_splitter'])
            
    def _choose_sampler(self):
        if self._pipeline_settings['oversampler'] == 'ros':
            self._sampler = RandomOversampler
        if self._pipeline_settings['oversampler'] == 'clus_ros':
            self._sampler = RandomClusterOversampler
        if self._pipeline_settings['oversampler'] == 'shuffle_ros':
            self._sampler = TimeSeriesShufflingOversampler
        if self._pipeline_settings['oversampler'] == 'weier':
            self._sampler = WeierOversampler
        if self._pipeline_settings['oversampler'] == 'none':
            self._sampler = NoSampler
        if self._pipeline_settings['oversampler'] == 'markov':
            self._sampler = MarkovOversampler
            
    def _choose_model(self):
        logging.debug('model: {}'.format(self._pipeline_settings['model']))
        if self._pipeline_settings['model'] == 'ts_attention':
            self._model = TimestepAttentionModel
            gs_path = './configs/gridsearch/gs_LSTM.yaml'
        if self._pipeline_settings['model'] == 'flipped_bilstm':
            self._model = FlippedBiLSTMModel
            gs_path = './configs/gridsearch/tobecreated'
        if self._pipeline_settings['model'] == 'tuglet_lstm':
            self._model = TugletLSTMModel
            gs_path = './configs/gridsearch/tobecreated'

    def _get_num_classes(self):
        if self._settings['experiment']['labels'] == 'binconcepts':
            self._n_classes = 2
        if self._settings['experiment']['labels'] == 'pass':
            self._n_classes = 2
        if self._settings['experiment']['labels'] == 'label':
            if 'binary' in self._settings['data']['feature']:
                self._n_classes = 2
            else:
                self._n_classes = 9
        self._settings['experiment']['n_classes'] = self._n_classes
    def _choose_scorer(self):
        self._get_num_classes()
        if self._n_classes == 2:
            self._scorer = BinaryClfScorer
        elif self._n_classes > 2:
            self._scorer = MultiClfScorer
            
    def _choose_gridsearcher(self):
        if self._pipeline_settings['gridsearch'] == 'none':
            ''
                
    def _choose_xvalidator(self):
        if self._pipeline_settings['crossvalidator'] == 'nonnested':
            self._choose_splitter(self._pipeline_settings['splitter'])
            self._gridsearch = {}
            self._xval = NonNestedRankingCrossVal(self._settings, self._splitter, self._sampler, self._model, self._scorer)
                
    def _build_pipeline(self):
        self._choose_inner_splitter()
        self._choose_outer_splitter()
        self._choose_gridsearch_splitter()
        self._choose_sampler()
        self._choose_model()
        self._choose_scorer()
        self._choose_xvalidator()
        
    def train(self, sequences:list, labels:list, demographics:list):
        results = self._xval.xval(sequences, labels, demographics)
        