import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree, rmtree
from ml.models.model import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Bidirectional, LSTM,Masking,Embedding
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate,train_test_split,GridSearchCV
from sklearn.preprocessing import normalize
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical


from numpy.random import seed

class FlippedBiLSTMModel(Model):
    """This class implements a BiLSTM as designed by Vinitra et al. in 
    'Evaluating the Explainers: Black-Box Explainable Machine Learning for Student Success Prediction in MOOCs'.
        Model (Model): inherits from the model class

    Paper link for the architecture: https://arxiv.org/pdf/2207.00551.pdf
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'flipped classroom bi-lstm'
        self._notation = 'fcbilstm'
        self._model_settings = settings['ml']['models']['flipped_classroom_bi_lstm']
        self._fold = 0

    def _format(self, x:list, y:list) -> Tuple[list, list]:
        #y needs to be one hot encoded
        features = np.array(x)
        features = features.reshape(features.shape[0], -1)
        return features, np.array(y)
    
    def _format_features(self, x:list) -> list:
        features = np.array(x)
        features = features.reshape(features.shape[0], -1)
        return features

    def load_model_weights(self, x:np.array, checkpoint_path:str):
        """Given a data point x, this function sets the model of this object

        Args:
            x ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        x = self._format_features(x) 
        self._init_model(x)
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['binary_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        checkpoint = tf.train.Checkpoint(self._model)
        temporary_path = '../experiments/temp_checkpoints/training/'
        if os.path.exists(temporary_path):
            rmtree(temporary_path)
            copytree(checkpoint_path, temporary_path, dirs_exist_ok=True)
        checkpoint.restore(temporary_path)
    
    def _get_csvlogger_path(self) -> str:
        csv_path = '../experiments/{}{}/{}/logger/{}/'.format(self._experiment_root, self._experiment_name, self._outer_fold, self._notation)
        csv_path += 'weektype{}_feat{}_course{}_ nweeks{}_nfeat{}'.format(
            self._model_settings['week_type'], self._model_settings['feature_type'], 
            self._model_settings['course'], self._model_settings['n_weeks'],
            self._model_settings['n_features']
        )
        os.makedirs(csv_path, exist_ok=True)
        checkpoint_path = csv_path + '/f{}_model_checkpoint'.format(self._gs_fold)
        csv_path += '/f' + str(self._gs_fold) + '_model_training.csv'
        return csv_path, checkpoint_path

    def _get_model_checkpoint_path(self) -> str:
        path = '../experiments/{}{}/{}/logger/{}/'.format(self._experiment_root, self._experiment_name, self._outer_fold, self._notation)
        path += 'weektype{}_feat{}_course{}_ nweeks{}_nfeat{}'.format(
            self._model_settings['week_type'], self._model_settings['feature_type'], 
            self._model_settings['course'], self._model_settings['n_weeks'],
            self._model_settings['n_features']
        )
        path += '/f{}_model_checkpoint/'.format(self._gs_fold)
        return path

    def _init_model(self, x:np.array):
        self._set_seed()
        n_dims = x.shape[0]
        look_back = 3
        # LSTM
        # define model
        self._model = Sequential()
        ###########Reshape layer################
        self._model.add(tf.keras.layers.Reshape(
            (self._model_settings['n_weeks'], self._model_settings['n_features']), 
            input_shape=(self._model_settings['n_weeks'] * self._model_settings['n_features'],))
            )

        self._model.add(Bidirectional(LSTM(self._model_settings['hidden_feature_num'][0], return_sequences=True)))
        self._model.add(Bidirectional(LSTM(self._model_settings['hidden_feature_num'][1])))
        self._model.add(Dense(1, activation='sigmoid'))

        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy', cce, auc])

        self._callbacks = []
        if self._model_settings['early_stopping']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, min_delta=0.001, 
                restore_best_weights=True
            )
            self._callbacks.append(early_stopping)

        # csv loggers
        csv_path, checkpoint_path = self._get_csvlogger_path()
        csv_logger = CSVLogger(csv_path, append=True, separator=';')
        self._callbacks.append(csv_logger)

        if self._model_settings['save_best_model']:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True)
            self._callbacks.append(model_checkpoint_callback)

        print(self._model.summary())

    def load_checkpoints(self, checkpoint_path:str, x:list):
        """Sets the inner model back to the weigths present in the checkpoint folder.
        Checkpoint folder is in the format "../xxxx_model_checkpoint/ and contains an asset folder,
        a variables folder, and index and data checkpoint files.

        Args:
            checpoint_path (str): path to the checkpoint folder
            x (list): partial sample of data, to format the layers
        """
        x = self._format_features(x) 
        self._init_model(x)
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy', cce, auc]
        )
        checkpoint = tf.train.Checkpoint(self._model)

        temporary_path = '../experiments/temp_checkpoints/training/'
        checkpoint.restore(checkpoint_path)

        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)

        # print(y_train)

        print(x_train.shape)
        self._init_model(x_train)
        self._history = self._model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=self._model_settings['batch_size'],
            shuffle=self._model_settings['shuffle'],
            epochs=self._model_settings['epochs'],
            verbose=self._model_settings['verbose'],
            callbacks=self._callbacks
        )

        if self._model_settings['save_best_model']:
            checkpoint_path = self._get_model_checkpoint_path()
            self.load_model_weights(x_train, checkpoint_path)
            self._best_epochs = np.argmax(self._history.history['val_auc'])
            print('best epoch: {}'.format(self._best_epochs))

        self._fold += 1
        
    def predict(self, x:list) -> list:
        xpredict = self._format_features(x)
        ypredict = self._model.predict(xpredict)
        ypredict = [1 if yy > 0.5 else 0 for yy in ypredict]
        return ypredict
    
    def predict_proba(self, x:list) -> list:
        xpredict = self._format_features(x)
        yproba = self._model.predict(xpredict)
        yproba = [[1 - yy, yy] for yy in yproba]
        return yproba
    
    def save(self) -> str:
        return self.save_tensorflow()
    
    def get_path(self, fold: int) -> str:
        return self.get_path(fold)
            
    def save_fold(self, fold: int) -> str:
        return self.save_fold_tensorflow(fold)

    def save_fold_early(self, fold: int) -> str:
        return self.save_fold_early_tensorflow(fold)
    
    
    
    
