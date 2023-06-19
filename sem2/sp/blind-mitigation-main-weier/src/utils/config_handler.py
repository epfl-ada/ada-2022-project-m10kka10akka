import os
import yaml
import string
from os import path as pth
from datetime import datetime

class ConfigHandler:
    def __init__(self, settings:dict):
        self._settings = settings
        
    def get_settings(self):
        return dict(self._settings)
        
    def get_oversample_experiment_name(self):
        """Creates the experiment name in the following path:
            '../experiments/experiment root/yyyy_mm_dd_index/'
            index being the first index in increasing order starting from 0 that does not exist yet.
            
            This function:
            - returns the experiment config name 
            - creates the folder with the right experiment name at ../experiments/experiment root/yyyy_mm_dd_index
            - dumps the config in the newly created folder

        Args:
            settings ([type]): read config

        Returns:
            [str]: Returns the name of the experiment in the format of 'yyyy_mm_dd_index'
        """
        path = '../experiments/{}/{}/{}/{}/{}/'.format(
            self._settings['experiment']['root_name'],
            self._settings['data']['dataset'],
            self._settings['data']['feature'],
            self._settings['ml']['pipeline']['model'],
            self._settings['data']['label']
        )
        today = datetime.today().strftime('%Y-%m-%d')
        today = today.replace('-', '_')
        starting_index = 0
        
        # first index
        experiment_name = '{}{}_{}/'.format(path, today, starting_index)
        while (pth.exists(experiment_name)):
            starting_index += 1
            experiment_name = '{}{}_{}/'.format(path, today, starting_index)
            
        self._experiment_path = experiment_name
        os.makedirs(self._experiment_path, exist_ok=True)
        
        experiment_name_path = '/{}/{}/{}/{}/{}_{}/'.format(
            self._settings['data']['dataset'],
            self._settings['data']['feature'],
            self._settings['ml']['pipeline']['model'],
            self._settings['data']['label'],
            today,
            starting_index
        )
        self._settings['experiment']['name'] = experiment_name_path
      