import os

import pandas as pd
from matplotlib import pyplot as plt

class Plotter:
    """Implements plotting recurring functions
    """

    def __init__(self, settings:dict):
        self._name = 'plotter'
        self._notation = 'pltr'
        self._settings = dict(settings)
        self._root_plot_path = '../experiments/{}/figures/'.format(self._settings['experiment']['root_name'])
        os.makedirs(self._root_plot_path, exist_ok=True)

    def _plot_single_boxplot(self, data:list, colour:str, position:float):
        """Plot a boxplot of the data point, in the given colour, at the given x position
        Usually done to show non nested cross validation scores

        Args:
            data (list<float>): data containing n datapoints 
            colour (str): colour of the boxplot
            position (float): x position of the boxplot
        """
        box_item = plt.boxplot(data, positions=[position], patch_artist=True)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(box_item[item], color='black', linewidth=1)
        plt.setp(box_item['medians'], color='black', linewidth=2)
        plt.setp(box_item["boxes"], color=colour, edgecolor='black', linewidth=1)
        plt.setp(box_item["fliers"], markeredgecolor=colour, markersize=7)

    def _plot_single_barplot(self, xs:list, ys:list, colour:str, label:str):
        """Plot a barplot where each bar is on xs, and y high

        Args:
            xs (list): where to place the bars
            ys (list): how high the bars are
            color (str): colour of the bars to plot
            label (str): label of the bars
        """
        plt.bar(xs, ys, width=self._settings['style']['bar_width'], color=colour, label=label)

    def _update_csv(self, csv_data: dict):
        """takes in csv data and updates the csv of interest with it

        Args:
            csv_data (dict): for each experiment, and each metric (one line per experiment + metric in the new csv) we have:
                - the unique key by which to identify the experiment 
                - oversampling mode (none, random, etc.)
                - oversampling_proportion (equally balanced, etc.)
                - oversampling_attribute (criterion on which we oversampled)
                - metric in which the results of that rows are computed
                - scores per demographics (mean + std)
                - full scores

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        csv_path = '../experiments/{}/{}'.format(self._settings['csv']['folder'], self._settings['csv']['file'])
        try:
            current_results = pd.read_csv(csv_path, sep='\t', index_col=0)
            current_columns = [col for col in current_results.columns]
        except FileNotFoundError:
            current_results = pd.DataFrame()
            current_columns = [
                'key', 'oversampling_model', 'oversampling_proportion', 'oversampling_attribute', 'metric'
            ]
            for col in current_columns:
                current_results[col] = ''

        for experiment in csv_data:
            for metric in csv_data[experiment]:
                
                current_key = csv_data[experiment][metric]['key']
                mask = (current_results['key'] == current_key) & (current_results['metric'] == metric)
                temp = current_results[mask]
                if len(temp) == 1:
                    for column in csv_data[experiment][metric]:
                        current_results.loc[mask, column] = csv_data[experiment][metric][column]
                else:
                    cur_row = dict(csv_data[experiment][metric])
                    cur_row['key'] = current_key
                    cur_row['metric'] = metric
                    current_results = current_results.append(cur_row, ignore_index=True)
                    for key in current_results['key']:
                        print(key)
        if self._settings['dump']:
            print('saved at {}'.format(csv_path))
            current_results.to_csv(csv_path, sep='\t')
        

    def _savefig(self, name:str, subfolder=''): 
        """Saves the figure in experiments/<name>/figures/<subfolder>/<name>.svg

        Args:
            subfolder (str): name of the subfolders in figures in which the result needs to be saved
            name (str): name of the image to save
        """
        path = '{}{}{}.svg'.format(
            self._root_plot_path,
            subfolder,
            name
        )
        plt.savefig(path, format='svg', bbox_inches='tight')