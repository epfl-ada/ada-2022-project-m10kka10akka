import os
import re

import numpy as np
import pandas as pd

from IPython.core.display import display
from visualisers.plotter import Plotter

class CSVViewer(Plotter):

    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'csv_plotter'
        self._notation = 'csvpltr'
        self._csv_settings = dict(settings['csv'])

        self._ascending_map = {
            'roc': True,
            'fn': False,
            'tp': True,
            'fp': False,
            'balanced_accuracy': True
         }

    def _read_csv(self):
        csv_path = '{}/{}'.format(
            self._csv_settings['folder'],
            self._csv_settings['file']
        )
        self._csv = pd.read_csv(csv_path, sep='\t', index_col=0)

        if len(self._csv_settings['filtering_columns']) > 0:
            include_re = re.compile('({})'.format('|'.join(self._csv_settings['filtering_columns'])))
            temp = [c for c in self._csv.columns]
            mask = [True if len(include_re.findall(temp[idx])) > 0 else False for idx in range(len(temp))]
            columns = self._csv_settings['fixed'] + [self._csv.columns[idx] for idx in range(len(temp)) if mask[idx]]

            self._csv = self._csv[columns]
            if self._settings['drop_std']:
                self._csv = self._drop_std(self._csv)
        if len(self._csv_settings['filtering_experiments']) > 0:
            for col in self._csv_settings['filtering_experiments']:
                self._csv = self._csv[self._csv[col].isin(self._csv_settings['filtering_experiments'][col])]


        # if self._settings['filtering']:
        #     include_re = re.compile('({})'.format('|'.join(self._settings['csv']['filtering_experiments'])))
        #     self._csv['temp'] = self._csv.apply(lambda row: '{}_{}_{}_{}'.format(
        #         row['key'], row['oversampling_model'], row['oversampling_proportion'], row['oversampling_attribute']
        #     ), axis=1)
        #     temp = list(self._csv['temp'])
        #     mask = [True if len(include_re.findall(temp[idx])) > 0 else False for idx in range(len(temp))]
        #     self._csv = self._csv[mask]

        #     include_re = re.compile('({})'.format('|'.join(self._settings['csv']['filtering_columns'])))
        #     temp = [c for c in self._csv.columns]
        #     mask = [True if len(include_re.findall(temp[idx])) > 0 else False for idx in range(len(temp))]
        #     columns = self._csv_settings['fixed'] + [self._csv.columns[idx] for idx in range(len(temp)) if mask[idx]]
        #     self._csv = self._csv[columns]
            if self._settings['drop_std']:
                self._csv = self._drop_std(self._csv)
        
        view_path = '{}/views_{}'.format(
            self._csv_settings['folder'],
            self._csv_settings['file'].replace('.tsv', '/')
        )
        os.makedirs(view_path, exist_ok=True)

    def _name_baseline(self, key:str):
        if self._csv_settings['baseline'] in key:
            return '0' + key
        else:
            return key

    def _is_baseline(self, key:str):
        if self._csv_settings['baseline'] in key:
            return True
        else:
            return False


    def _sort(self):
        if 'baseline' in self._settings['csv']['sorting']:
            self._csv['baseline'] = self._csv['key'].apply(lambda x: self._name_baseline(x))
        else:
            self._csv['baseline'] = self._csv['key']

        sorting_values = [col for col in self._csv_settings['sorting']]
        self._csv = self._csv.sort_values(sorting_values)

    def _colour_cell_baseline(self, columns:list, baseline:dict, row:dict, ascending:bool) -> str:
        colours = []
        for i_c, col in enumerate(columns):
            if col in self._csv_settings['fixed']:
                colours.append("background-color: white; text_wrap: True")
            else:
                if row['baseline']:
                    colours.append('background-color: plum; text_wrap: True')
                else:
                    if ascending and 'equalodds' not in col:
                        if baseline[col] < row[col]:
                            colours.append('background-color: darkkhaki; text_wrap: True')
                        else:
                            colours.append('background-color: white; text_wrap: True')
                    else:
                        if baseline[col] > row[col]:
                            colours.append('background-color: darkkhaki; text_wrap: True')
                        else:
                            colours.append('background-color: white; text_wrap: True')

        return colours 

    def _colour_single_baseline(self):
        self._csv['baseline'] = self._csv['key'].apply(lambda x: self._is_baseline(x))
        
        display_tables = {}
        for metric, metric_group in self._csv.groupby('metric'):
            group_df = metric_group.copy()

            baseline_df = group_df[group_df['baseline']]
            assert len(baseline_df) == 1

            non_baseline_df = group_df[~group_df['baseline']]
            
            group_df = baseline_df.append(non_baseline_df)

            group_df = group_df.style.apply(
                lambda row: self._colour_cell_baseline(
                group_df.columns,
                baseline_df.iloc[0], 
                row, 
                self._ascending_map[metric]
            ), axis=1
            )

            csv_path = '{}/views_{}/{}.xlsx'.format(
                self._csv_settings['folder'],
                self._csv_settings['file'].replace('.tsv', '/'),
                metric,
            )
            print(csv_path)
            group_df.to_excel(csv_path)
            display_tables[metric] = group_df
            if self._settings['display']:
                display(group_df)

        return display_tables

    def _drop_std(self, df):
        for col in df.columns:
            if 'std' in col:
                df = df.drop(col, axis=1)
        return df

    def view_csv(self):
        self._read_csv()
        if self._settings['sort']:
            self._sort()
        if self._settings['colour']:
            return self._colour_single_baseline()





            
            


        

    