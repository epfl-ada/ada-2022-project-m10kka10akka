import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

from ml.crossvalidators.crossvalidator import CrossValidator
from ml.splitters.splitter import Splitter
from ml.samplers.sampler import Sampler
from ml.models.model import Model
from ml.scorers.scorer import Scorer

class NonNestedRankingCrossVal(CrossValidator):
    """Implements non nested cross validation: 
            For each fold, get train and test set:
                Train your model on the train set
                Test your model with the test set
    Args:
        CrossValidator (CrossValidators): Inherits from the model class
    """
    
    def __init__(self, settings:dict, splitter: Splitter, sampler:Sampler, model:Model, scorer:Scorer):
        super().__init__(settings, model, scorer)
        self._name = 'nonnested cross validator'
        self._notation = 'nonnested_cval'

        self._splitter = splitter(settings, settings['ml']['nfolds']['full'])
        self._sampler = sampler(settings)
        self._fairness_metrics = self._settings['ml']['scorer']['fairness_metrics']
        
        
    def xval(self, sequences:list, labels:list, demographics:dict) -> dict:
        results = {}
        results['x'] = sequences
        results['y'] = labels
        results['demographics'] = demographics
        results['optim_scoring'] = self._xval_settings['optim_scoring'] 
        logging.debug('x:{}, y:{}'.format(sequences, labels))


        for f, (train_index, test_index) in enumerate(self._splitter.split(sequences, demographics['stratifier_col'])):
            print(' test index: {}'.format(test_index[0:5]))
            logging.debug('    length train: {}, length test: {}'.format(len(train_index), len(test_index)))
            logging.debug('    outer fold: {}'.format(f))
            logging.info('- ' * 30)
            logging.info('  Fold {}'.format(f))
            logging.debug('    train indices: {}'.format(train_index))
            logging.debug('    test indices: {}'.format(test_index))
            
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['test_index'] = test_index

            # division train / test
            x_train = [sequences[xx] for xx in train_index]
            y_train = [labels[yy] for yy in train_index]
            oversampler_train = [demographics['oversampler_col'][tidx] for tidx in train_index]
            demographics_train = {}
            for demo in demographics:
                demographics_train[demo] = [demographics[demo][idx] for idx in train_index]
            x_test = [sequences[xx] for xx in test_index]
            y_test = [labels[yy] for yy in test_index]
            
            # Inner loop
            x_resampled, y_resampled, idx_resampled = self._sampler.sample(x_train, oversampler_train, y_train, demographics_train)
            results[f]['oversample_indexes'] = idx_resampled
            
            logging.debug('  * data format: x [{}], y [{}]'.format(np.array(x_resampled).shape, np.array(y_resampled).shape))
            
            
            model = self._model(self._settings)
            if model.get_settings()['save_best_model']:
                train_x, val_x, train_y, val_y = train_test_split(
                    x_resampled, y_resampled, 
                    test_size=0.1, random_state=self._settings['seeds']['splitter']
                )
                results[f]['model_train_x'] = train_x
                results[f]['model_train_y'] = train_y
                results[f]['model_val_x'] = val_x
                results[f]['model_val_y'] = val_y
            else:
                train_x, train_y = x_resampled, y_resampled
                val_x, val_y = x_test, y_test

            model.set_outer_fold(f)
            model.fit(train_x, train_y, x_val=val_x, y_val=val_y)
            results[f]['x_resampled'] = x_resampled
            results[f]['y_resampled'] = y_resampled
            results[f]['x_resampled_train'] = train_x
            results[f]['y_resampled_train'] = train_y
            results[f]['x_resampled_val'] = val_x
            results[f]['y_resampled_val'] = val_y
            results[f]['best_params'] = model.get_settings()

            if model.get_settings()['save_best_model']:
                results[f]['best_epochs'] = model.get_best_epochs()

            model.save_fold(f)

            # Predict
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test, y_pred, y_proba)
            for id_d in demographics.keys():
                if '_col' not in id_d:
                    test_demo = [demographics[id_d][idx] for idx in test_index]
                    fairness_results = self._scorer.get_fairness_scores(y_test, y_pred, y_proba, test_demo, self._fairness_metrics)
                    results[f][id_d] = fairness_results
            logging.debug('    predictions: {}'.format(y_pred))
            logging.debug('    probability predictions: {}'.format(y_proba))
            
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f].update(test_results)
            
            print('Best Results on outer fold: {}'.format(test_results))
            logging.info('Best Results on outer fold: {}'.format(test_results))
            self._model_notation = model.get_notation()
            self.save_results(results)
        return results
    
    def save_results(self, results):
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/results/' 
        os.makedirs(path, exist_ok=True)
        
        path += self._notation + '_m' + self._model_notation + '_l' + str(self._settings['data']['adjuster']['limit']) + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            