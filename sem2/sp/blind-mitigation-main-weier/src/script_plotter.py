import os
import yaml
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from utils.config_handler import ConfigHandler
from visualisers.nonnested_plotter import NonNestedPlotter as redundant
from visualisers.compute_nonnested_plotter import NonNestedPlotter 
from visualisers.inverse_compute_nonnested_plotter import InverseNonNestedPlotter
from visualisers.csv_viewer import CSVViewer

def nonnested(settings):
    plotter = NonNestedPlotter(settings)
    plotter.test(settings)

def inverse_nonnested(settings):
    plotter = InverseNonNestedPlotter(settings)
    plotter.test(settings)

def view_csv(settings):
    viewer = CSVViewer(settings)
    viewer.view_csv()

def test(settings):
    print('no test')

    
def main(settings):
    if settings['inverse'] and settings['nonnested']:
        inverse_nonnested(settings)
    elif settings['nonnested']:
        nonnested(settings)

    if settings['csv_viewer']:
        view_csv(settings)
    if settings['test']:
        test(settings)

if __name__ == '__main__': 
    with open('./configs/plotter_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description='Plot the results')

    # Tasks
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    parser.add_argument('--nonnested', dest='nonnested', default=False, action='store_true')

    parser.add_argument('--csvviewer', dest='csv_viewer', default=False, action='store_true')

    # Plot Type
    parser.add_argument('--boxplot', dest='boxplot', default=False, action='store_true')
    parser.add_argument('--barplot', dest='barplot', default=False, action='store_true')
    parser.add_argument('--fairness', dest='fairness', default=False, action='store_true')
    parser.add_argument('--equalodds', dest='equal_odds', default=False, action='store_true')

    # CSV types
    parser.add_argument('--sort', dest='sort', default=False, action='store_true')
    parser.add_argument('--colour', dest='colour', default=False, action='store_true')
    parser.add_argument('--dropstd', dest='drop_std', default=False, action='store_true')
    parser.add_argument('--filtering', dest='filtering', default=False, action='store_true')


    # computation mode
    parser.add_argument('--overall', dest='overall', default=False, action='store_true')
    parser.add_argument('--combined', dest='combined', default=False, action='store_true')
    parser.add_argument('--fold', dest='fold', default=False, action='store_true')
    parser.add_argument('--inverse', dest='inverse', default=False, action='store_true')
    parser.add_argument('--pairwise', dest='pairwise', default=False, action='store_true')

    # Arguments
    parser.add_argument('--show', dest='show', default=False, action='store_true') # image
    parser.add_argument('--save', dest='save',  default=False, action='store_true') # image
    parser.add_argument('--dump', dest='dump',  default=False, action='store_true') # csv
    parser.add_argument('--print', dest='print', default=False, action='store_true') # csv
    parser.add_argument('--display', dest='display', default=False, action='store_true') # csv


    
    settings.update(vars(parser.parse_args()))
    main(settings)