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
    if settings['mode'] == 'baseline': 
        settings['ml']['pipeline']['oversampler'] = 'none'
    elif settings['mode'] == 'os':
        settings['ml']['pipeline']['oversampler'] = 'ros'
        settings['ml']['oversampler']['rebalancing_mode'] = 'cascade'


    settings['ml']['oversampler']['oversampling_col'] = settings['attributes'].split('.')


    oversampling_attributes = '_'.join(settings['ml']['oversampler']['oversampling_col'])
    settings['experiment']['root_name'] += '/{}_oversampling/{}'.format(settings['ml']['oversampler']['rebalancing_mode'], oversampling_attributes)


    settings['experiment']['labels'] = 'binconcepts'
    settings['data']['dataset'] = 'beerslaw'
    settings['data']['feature'] = 'simplestates_cluster'
    settings['data']['label'] = 'binconcepts'
    settings['data']['others']['gender'] = ['3', '4']
    settings['data']['adjuster']['limit'] = 300
    settings['ml']['pipeline']['model'] = 'ts_attention'
    settings['ml']['splitter']['stratifier_col'] = ['binvector']

    return settings


def main(settings):
    settings = _process_arguments(settings)
    oversamplesimple(settings)
    # if settings['test']:
    #     test(settings)

if __name__ == '__main__': 
    with open('./configs/aayush_config.yml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description='Plot the results')

    # Tasks
    parser.add_argument('--mode', dest='mode', default='.', action='store', help='list of the criteria by which to oversample, separated by dots: gender.age')
    parser.add_argument('--attributes', dest='attributes', default='.', action='store', help='list of the criteria by which to oversample, separated by dots: gender.age')
    # dataset
    parser.add_argument('--simulation', dest='simulation', default=False, action='store', help='what data to use out of: tuglet, flipped or beer')

    
    settings.update(vars(parser.parse_args()))
    main(settings)

def run_script(settings):
    # run the baseline:
    "$python script_aayush.py --mode baseline"

    # run the model with simple oversampling (rebalancing the labels)
    "$python script_aayush.py --mode os --attributes gender.year.language.field"
