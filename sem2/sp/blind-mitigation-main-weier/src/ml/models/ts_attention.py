import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree, rmtree
from ml.models.model import Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model as Mod
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import get as get_loss, Loss
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.metrics import get as get_metric, Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import initializers
from keras import regularizers
from keras import constraints
from tensorflow.keras.layers import Layer
from keras import backend as K

from numpy.random import seed

class TimestepAttentionModel(Model):
    """This class implements an LSTM with attention where the last timestamp of the lstm layer is concatenated with the attention output
    Args:
        Model (Model): inherits from the model class

    Notion link to architecture:
        https://www.notion.so/Option-1-bb6797b337064730be21b1a19dae15f5
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'last timestep attention'
        self._notation = 'ltsatt'
        self._model_settings = settings['ml']['models']['ts_attention']
        self._maxlen = self._settings['data']['adjuster']['limit']
        self._fold = 0

    def _format(self, x:list, y:list) -> Tuple[list, list]:
        #y needs to be one hot encoded
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        y_vector = to_categorical(y, num_classes=self._n_classes)
        return x_vector, y_vector
    
    def _format_features(self, x:list) -> list:
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        return x_vector

    def _format_prior_features(self, x):
        priors = x[:, :, :self._prior_states]
        features = x[:, :, self._prior_states:]
        return priors, features

    def _get_rnn_layer(self, return_sequences:bool, l:int):
        n_cells = self._model_settings['n_cells'][l]
        if self._model_settings['cell_type'] == 'LSTM':
            layer = layers.LSTM(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'GRU':
            layer = layers.GRU(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'RNN':
            layer = layers.SimpleRNN(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'BiLSTM':
            layer = layers.LSTM(units=n_cells, return_sequences=return_sequences)
            layer = layers.Bidirectional(layer=layer)
        return layer

    def self_attention(self, x):
        ''' 
        from github repo: https://github.com/declare-lab/multimodal-deep-learning/blob/main/contextual-multimodal-fusion/trimodal_attention_models.py
        .  stands for dot product 
        *  stands for elemwise multiplication
            
        m = x . transpose(x)
        n = softmax(m)
        o = n . x  
        a = o * x           
        
        return a
        '''
        m = layers.dot([x, x], axes=[2,2])
        n = layers.Activation('softmax')(m)
        o = layers.dot([n, x], axes=[2,1])
        a = layers.multiply([o, x])
        return a

    def _get_csvlogger_path(self) -> str:
        csv_path = '../experiments/{}{}/{}/logger/{}/'.format(self._experiment_root, self._experiment_name, self._outer_fold, self._notation)
        csv_path += 'ct{}_nlayers{}_ncells{}_flatten{}'.format(
            self._model_settings['cell_type'], self._model_settings['n_layers'], self._model_settings['n_cells'], self._model_settings['flatten']
        )
        csv_path += '_drop{}_optim{}_loss{}_bs{}_ep{}'.format(
            self._model_settings['dropout'], self._model_settings['optimiser'], self._model_settings['loss'], self._model_settings['batch_size'], self._model_settings['epochs']
        )
        os.makedirs(csv_path, exist_ok=True)
        checkpoint_path = csv_path + '/f{}_model_checkpoint'.format(self._gs_fold)
        csv_path += '/f' + str(self._gs_fold) + '_model_training.csv'
        return csv_path, checkpoint_path

    def _get_model_checkpoint_path(self) -> str:
        path = '../experiments/{}{}/{}/logger/{}/'.format(self._experiment_root, self._experiment_name, self._outer_fold, self._notation)
        path += 'ct{}_nlayers{}_ncells{}_flatten{}'.format(
            self._model_settings['cell_type'], self._model_settings['n_layers'], self._model_settings['n_cells'], self._model_settings['flatten']
        )
        path += '_drop{}_optim{}_loss{}_bs{}_ep{}'.format(
            self._model_settings['dropout'], self._model_settings['optimiser'], self._model_settings['loss'], self._model_settings['batch_size'], self._model_settings['epochs']
        )
        path += '/f{}_model_checkpoint/'.format(self._gs_fold)
        return path

    def _retrieve_attentionlayer(self):
        return self._model.layers[4]

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
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        # print('pre-weight check: {}'.format(self._model.layers[2].weights[0][0]))
        checkpoint = tf.train.Checkpoint(self._model)

        print(checkpoint_path)
        # checkpoint_path += '/variables'
        temporary_path = '../experiments/temp_checkpoints/training/'
        if os.path.exists(temporary_path):
            rmtree(temporary_path)
            copytree(checkpoint_path, temporary_path, dirs_exist_ok=True)
        checkpoint.restore(temporary_path)
        # print('post-weight check: {}'.format(self._model.layers[2].weights[0][0]))

    def _init_model(self, x:np.array):
        print('Initialising prior model')
        self._set_seed()
        input_layer = layers.Input(shape=(x.shape[1], x.shape[2]), name='input_prior')
        full_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_prior')(input_layer)

        selfattention_features = self.self_attention(full_features)
        print('       sa {}'.format(selfattention_features.shape)) # expect shape of None, 900, 10
        print('       ff {}'.format(full_features.shape)) # expect shape of 900

        concatenated = layers.Concatenate(axis=2)([full_features, selfattention_features])
        print('       concatenated {}'.format(concatenated.shape)) # expect shape of 900 x (lstm cells + 1)

        for l in range(int(self._model_settings['n_layers']) -1):
            concatenated = self._get_rnn_layer(return_sequences=True, l=l)(concatenated)
        concatenated = self._get_rnn_layer(return_sequences=False, l=self._model_settings['n_layers'] - 1)(concatenated)

        classification_layer = layers.Dense(self._settings['experiment']['n_classes'], activation='softmax')(concatenated)

        self._model = Mod(input_layer, classification_layer)
        self._inference_model = Mod(input_layer, selfattention_features)

        # compiling
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        
        # callbacks
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
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        checkpoint = tf.train.Checkpoint(self._model)

        temporary_path = '../experiments/temp_checkpoints/training/'
        checkpoint.restore(checkpoint_path)

        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)

        self._init_model(x_train)
        print('EPOCHS', self._model_settings['epochs'],)
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
        return self.predict_tensorflow(x)
    
    def predict_proba(self, x:list) -> list:
        return self.predict_proba_tensorflow(x)
    
    def save(self) -> str:
        return self.save_tensorflow()
    
    def get_path(self, fold: int) -> str:
        return self.get_path(fold)
            
    def save_fold(self, fold: int) -> str:
        return self.save_fold_tensorflow(fold)

    def save_fold_early(self, fold: int) -> str:
        return self.save_fold_early_tensorflow(fold)
    
    
    
    
