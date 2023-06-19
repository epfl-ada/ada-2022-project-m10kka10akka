import os
import yaml
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from utils.config_handler import ConfigHandler
from data_handlers.data_loader import DataLoader
from ml.xval_maker import XValMaker

def oversamplesimple(settings):
    ch = ConfigHandler(settings)
    ch.get_oversample_experiment_name()

    print(settings['experiment'])

    dl = DataLoader(settings)
    sequences, labels, demographics = dl.load_data()
    xval = XValMaker(settings)
    xval.train(sequences, labels, demographics)

    config_path = '../experiments/' + settings['experiment']['root_name'] + settings['experiment']['name'] + '/config.yaml'
    with open(config_path, 'wb') as fp:
        pickle.dump(settings, fp)

def test(settings):
    print('no test')

def _process_arguments(settings):
    # Oversampling
    if settings['oversampler'] != '':
        settings['ml']['pipeline']['oversampler'] = settings['oversampler']
        
    if settings['equal_balancing']: 
        settings['ml']['oversampler']['rebalancing_mode'] = 'equal_balancing'
    elif settings['majority_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'major'
        settings['ml']['oversampler']['oversampling_factor'] = 1.5
    elif settings['majority_only_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'major_only'
        settings['ml']['oversampler']['oversampling_factor'] = 1.5
    elif settings['minority_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'minor'
        settings['ml']['oversampler']['oversampling_factor'] = 1
    elif settings['minority_only_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'minor_only'
        settings['ml']['oversampler']['oversampling_factor'] = 1
    elif settings['cascade_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'cascade'
        settings['ml']['oversampler']['oversampling_factor'] = 1
    elif settings['withingroup_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'within_group'
        settings['ml']['oversampler']['oversampling_factor'] = 1
    elif settings['cascadewithingroup_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'cascadewithin_group'
        settings['ml']['oversampler']['oversampling_factor'] = 1
    elif settings['majorwithingroup_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'majorwithin_group'
        settings['ml']['oversampler']['oversampling_factor'] = 1.5
    elif settings['minorwithingroup_balancing']:
        settings['ml']['oversampler']['rebalancing_mode'] = 'minorwithin_group'
        settings['ml']['oversampler']['oversampling_factor'] = 1

    if settings['oversampling_attribute'] != '.':
        oversampling_attributes = settings['oversampling_attribute'].split('.')
        settings['ml']['oversampler']['oversampling_col'] = [att for att in oversampling_attributes]

    oversampling_attributes = '_'.join(settings['ml']['oversampler']['oversampling_col'])
    settings['experiment']['root_name'] += '/{}_oversampling/{}'.format(settings['ml']['oversampler']['rebalancing_mode'], oversampling_attributes)

    # Data
    if settings['simulation'] == 'flipped':
        settings['experiment']['labels'] = 'pass'
        settings['data']['dataset'] = 'flipped'
        settings['data']['feature'] = 'cluster_demo_features'
        settings['data']['label'] = 'pass'
        settings['ml']['pipeline']['model'] = 'flipped_bilstm'
        settings['ml']['splitter']['stratifier_col'] = ['pass']
        settings['ml']['oversampler']['within_group'] = 'cluster_aied_paola'

    if settings['simulation'] == 'beer':
        settings['experiment']['labels'] = 'binconcepts'
        settings['data']['dataset'] = 'beerslaw'
        settings['data']['feature'] = 'simplestates_cluster'
        settings['data']['label'] = 'binconcepts'
        settings['data']['others']['gender'] = ['3', '4']
        settings['data']['adjuster']['limit'] = 300
        settings['ml']['pipeline']['model'] = 'ts_attention'
        settings['ml']['splitter']['stratifier_col'] = ['binvector']

    if settings['simulation'] == 'tuglet':
        settings['experiment']['labels'] = 'label'
        settings['data']['dataset'] = 'tuglet'
        settings['data']['feature'] = 'bilal_allincluded_binary'
        settings['data']['label'] = 'label'
        settings['ml']['pipeline']['model'] = 'tuglet_lstm'
        settings['ml']['splitter']['stratifier_col'] = ['label']
        settings['ml']['models']['tuglet_lstm']['score'] = 2
        settings['ml']['oversampler']['within_group'] = 'cluster'

    if settings['simulation'] == 'tugletbinary':
        settings['experiment']['labels'] = 'label'
        settings['data']['dataset'] = 'tuglet'
        settings['data']['feature'] = 'bilal_allincluded_binary'
        settings['data']['label'] = 'label'
        settings['ml']['pipeline']['model'] = 'tuglet_lstm'
        settings['ml']['splitter']['stratifier_col'] = ['label']
        settings['ml']['models']['tuglet_lstm']['score'] = 2

    return settings


def main(settings):
    settings = _process_arguments(settings)
    if settings['oversamplesimple']:
        oversamplesimple(settings)
    if settings['test']:
        test(settings)

if __name__ == '__main__': 
    with open('./configs/oversample_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description='Plot the results')

    # Tasks
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    parser.add_argument('--oversamplesimple', dest='oversamplesimple', default=False, action='store_true')
    parser.add_argument('--oversampler', dest='oversampler', default='', action='store_true', help='what oversampler to use: ros, none')

    # oversampling attributes
    parser.add_argument('--oversamplingatt', dest='oversampling_attribute', default='.', action='store', help='list of the criteria by which to oversample, separated by dots: gender.age')
    parser.add_argument('--equal', dest='equal_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--majority', dest='majority_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--majorityonly', dest='majority_only_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--minority', dest='minority_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--minorityonly', dest='minority_only_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--cascade', dest='cascade_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--within', dest='withingroup_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--cascadewithin', dest='cascadewithingroup_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--majorwithin', dest='majorwithingroup_balancing', default=False, action='store_true', help='oversampling with equal balancing')
    parser.add_argument('--minorwithin', dest='minorwithingroup_balancing', default=False, action='store_true', help='oversampling with equal balancing')

    # dataset
    parser.add_argument('--simulation', dest='simulation', default=False, action='store', help='what data to use out of: tuglet, flipped or beer')

    
    settings.update(vars(parser.parse_args()))
    main(settings)