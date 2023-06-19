import os
import re
from unittest import result
import yaml
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ml.scorers.binary_scorer import BinaryClfScorer
from visualisers.plotter import Plotter

class NonNestedPlotter(Plotter):

    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'non nested plotter'
        self._notation = 'nnpltr'

        self._figwidth = self._settings['style']['figsize_width']
        self._figheight = self._settings['style']['figsize_height']


    def _get_colour(self, n:int):
        """Returns n colours from the chosen palette

        Args:
            n (int): number of colours wanted

        Returns:
            list: n randomly chosen colours. If n > len(palette) -> some colours will be selected more than
            once
        """
        colours = [
            '#241A7A', '#7A89F7', '#AFB9FA', '#EFF3F6', '#EF8C34',
            '#594DD4', '#213AF2', '#A8BCC7', '#A8BCC7', '#A8BCC7', 
            '#618595', '#618595'
        ]
        replace_bool = n > len(colours)
        return np.random.choice(colours, size=n, replace=replace_bool)

    def _crawl_files(self):
        """Crawls the files in the experiment root name folder as given by the config files

        Returns:
            results_files (list): list of all the results files for that experiment
            config_files (list): list of all the config results for that experiment
        """
        results_files = []
        experiment_path = '../experiments/{}'.format(self._settings['experiment']['root_name'])
        for (dirpath, dirnames, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            results_files.extend(files)
        configs_files = [f for f in results_files if 'config.yaml' in f]
        results_files = [f for f in results_files if 'results' in f]
        for e in self._settings['experiment']['exclude']:
            configs_files = [f for f in configs_files if e not in f]
            results_files = [f for f in results_files if e not in f]
        assert len(results_files) == len(configs_files)
        return results_files, configs_files

    def _load_data(self, results_files:list, configs_files:list):
        """Takes the regex expressions in the plotter config file and uses it to name the different
        subexperiments, then loads the results files and the config file

        Args:
            results_files (list): list of all the results files for that experiment
            config_files (list): list of all the config results for that experiment

        Returns:
            results (dict): 
                experiment label: config_file, results
        """
        results = {}
        regexes = []
        for r in self._settings['experiment']['regexes']:
            regexes.append(re.compile(r))

        for result in results_files:
            key = []
            try:
                for r in regexes:
                    key.append(r.findall(result)[0]) 
                cs = [c for c in configs_files]
                for k in key:
                    cs = [c for c in cs if k in c]

                key = '_'.join(key)
                with open(result, 'rb') as fp:
                    result_data = pickle.load(fp)
                with open(cs[0], 'rb') as fp:
                    config_data = pickle.load(fp)
                results[key] = {
                    'result': result_data,   
                    'config': config_data
                }
            except IndexError:
                print('file {} will not be processed'.format(result))

        return results

########### Get Fairness Metrics
    def _get_scorer(self, ytrue:list):
        """Looks into the number of labels from the classification problem in order to determine
        whether the scores should be computed in a binary or mutlipliscinary fashion.

        Args:
            results (dict): results dictionary as recorded by the  non-nested cross validatino class

        Returns:
            Scorer: object to compute different scores
        """
        new_settings = dict(self._settings)
        new_settings['ml'] = {
            'scorer': {'scoring_metrics' : self._settings['data']['fairness_metrics']}
        }
        if len(np.unique(ytrue)) == 2:
            return BinaryClfScorer(new_settings)
    
    def _get_fairness_data(self, resultsconfig:dict):
        """Retrieves the ypred, yprobs, ytrue and according demographics.

        Args:
            results (dict): _description_
        """
        y_trues = {}
        y_preds = {}
        y_probs = {}
        demogs = {k: {} for k in self._settings['data']['demographics']}
        
        for experiment in resultsconfig:
            e = 0
            fold = 0
            results = resultsconfig[experiment]['result']
            y_true = []
            y_pred = []
            y_prob = []
            demo_list = {k: [] for k in self._settings['data']['demographics']}
            while(fold in results):
                y_true = y_true + [results['y'][idx] for idx in results[fold]['test_index']]
                y_pred = y_pred + results[fold]['y_pred']
                y_prob = y_prob + [yy for yy in results[fold]['y_proba']]
                for demo in demo_list:
                    demo_list[demo] = demo_list[demo] + [results['demographics'][demo][idx] for idx in results[fold]['test_index']]
                fold += 1
            
            y_trues[experiment] = y_true
            y_preds[experiment] = y_pred
            y_probs[experiment] = y_probs
            demogs[experiment] = demo_list

        return y_trues, y_preds, y_probs, demogs

    def _get_onemetric_onedemographics_overall_barplots(self, ytrues:list, ypreds:list, yprobs:list, demographics:list):
        scorer = self._get_scorer(ytrues[list(ytrues.keys())[0]])
        metrics = self._settings['data']['fairness_metrics']
        colours = self._get_colour(len(ytrues))


        for demographic_type in self._settings['data']['demographics']:
            for metric in self._settings['data']['fairness_metrics']:
                plt.figure(figsize=(self._figwidth, self._figheight))
                plt.title('{} for {}'.format(metric, demographic_type))
                e = 0

                demo_attributes = np.unique(demographics[list(ytrues.keys())[0]][demographic_type])
                for i_e, experiment in enumerate(ytrues):
                    x = e
                    xs = []
                    ys = []
                    labels = []
                    ytrue = ytrues[experiment]
                    ypred = ypreds[experiment]
                    yprob = yprobs[experiment]
                    demo = demographics[experiment]
                    scores = scorer.get_fairness_scores(ytrue, ypred, yprob, demo[demographic_type], [metric])
                
                    for d in demo_attributes:
                        if scores[metric][d] == -1:
                            print('undefined {} for demographic {} for all folds {}'.format(metric, d))
                            continue
                        xs.append(x)
                        ys.append(scores[metric][d])
                        labels.append(d)

                        x += (self._settings['style']['bar_width'] * len(ytrues)) + self._settings['style']['groupspacing']

                    
                    self._plot_single_barplot(xs, ys, colours[i_e], experiment)
                    e += self._settings['style']['bar_width'] + self._settings['style']['xspacing']
                
                xs = [
                    (self._settings['style']['bar_width']+self._settings['style']['xspacing']) * (len(ytrues)-1)/2 +
                    i * (self._settings['style']['bar_width'] * len(ytrues) + self._settings['style']['groupspacing'])
                    for i in range(len(demo_attributes))
                    ]
                plt.xticks(xs, demo_attributes)

                plt.legend()
                # if self._settings['print']:
                    # print('metric: {}')
                    # print_df = pd.DataFrame()
                    # print_df['experiments'] = 
                if self._settings['save']:
                    self._savefig('{}_{}_overall_barplot'.format(metric, demographic_type), 'fairness_metrics')
                if self._settings['show']:
                    plt.show()
                else:
                    plt.close()

    def _get_onemetric_onedemographics_fold_boxplots(self, results:dict):
        colours = self._get_colour(len(results))
        for demographic_type in self._settings['data']['demographics']:
            for metric in self._settings['data']['fairness_metrics']:
                plt.figure(figsize=(self._figwidth, self._figheight))
                plt.title('{} for {}'.format(metric, demographic_type))
                e = 0

                experiments_details = {}
                for i_e, experiment in enumerate(results):
                    experiments_details[experiment] = {}
                    x = e
                    
                    xs = []
                    ys = []
                    labels = []
                    i_fold = 0
                    demo_scores = {r_key: [] for r_key in results[experiment]['result'][i_fold][demographic_type][metric]}
                    for d in demo_scores:
                        i_fold = 0
                        fold_scores = []
                        while (i_fold in results[experiment]['result']):
                            if d not in results[experiment]['result'][i_fold][demographic_type][metric]:
                                i_fold += 1
                                continue
                            if results[experiment]['result'][i_fold][demographic_type][metric][d] == -1:
                                i_fold += 1
                                print('undefined metric for fold {}'.format(i_fold - 1))
                                continue
                            fold_scores.append(results[experiment]['result'][i_fold][demographic_type][metric][d])
                            i_fold += 1

                        experiments_details[experiment]['{}_mean'.format(d)] = np.mean(fold_scores)
                        experiments_details[experiment]['{}_std'.format(d)] = np.std(fold_scores)
                        self._plot_single_boxplot(fold_scores, colours[i_e], x)
                        # plt.scatter([x for _ in fold_scores], fold_scores, color='black')
                        x += (self._settings['style']['bar_width'] * len(results)) + self._settings['style']['groupspacing']

                    
                    e += self._settings['style']['bar_width'] + self._settings['style']['xspacing']
                
                # Style
                xs = [
                    (self._settings['style']['bar_width']+self._settings['style']['xspacing']) * (len(results)-1)/2 +
                    i * (self._settings['style']['bar_width'] * len(results) + self._settings['style']['groupspacing'])
                    for i in range(len(demo_scores))
                    ]
                plt.xticks(xs, demo_scores.keys())
                plt.yticks(np.arange(0, 1, 0.1))
                plt.grid()
                plt.ylim([0, 1])
                x -= (self._settings['style']['bar_width'] * len(results)) + self._settings['style']['groupspacing']
                plt.xlim([-self._settings['style']['xmargins'], x + self._settings['style']['xmargins']])

                for i_e, experiment in enumerate(results):
                    plt.plot([0, 1], [-2, -2], color=colours[i_e], label=experiment)
                plt.legend()

                if self._settings['print']:
                    print()
                    print('*' * 40)
                    print('demographics: {}, metric: {}'.format(demographic_type, metric))
                    print_df = pd.DataFrame(experiments_details)
                    print(print_df)
                    print('*' * 40)
                    print()
                if self._settings['save']:
                    self._savefig('{}_{}_fold_boxplot'.format(metric, demographic_type), 'fairness_metrics')
                if self._settings['show']:
                    plt.show()
                else:
                    plt.close()

    def _get_fairness_barplots(self, resultconfigs:dict):
        if self._settings['overall']:
            y_trues, y_preds, y_probs, demographics = self._get_fairness_data(resultconfigs)
            self._get_onemetric_onedemographics_overall_barplots(y_trues, y_preds, y_probs, demographics)
        elif self._settings['fold']:
            self._get_onemetric_onedemographics_fold_boxplots(resultconfigs)

#### Compute equal odds
    def _get_onemetric_onedemographics_fold_equal_odds(self, results:dict):
        for metric in ['fp', 'tp']:
            for demographic_type in self._settings['data']['demographics']:
                experiments_details = {}
                for i_e, experiment in enumerate(results):
                    experiments_details[experiment] = {}
                    i_fold = 0
                    demo_scores = [r_key for r_key in results[experiment]['result'][i_fold][demographic_type][metric]]
                    for d in demo_scores:
                        i_fold = 0
                        fold_scores = []
                        while (i_fold in results[experiment]['result']):
                            if d not in results[experiment]['result'][i_fold][demographic_type][metric]:
                                i_fold += 1
                                continue
                            if results[experiment]['result'][i_fold][demographic_type][metric][d] == -1:
                                print('undefined {} for {} at fold {}'.format(metric, d, i_fold))
                                i_fold += 1
                                continue
                            fold_scores.append(results[experiment]['result'][i_fold][demographic_type][metric][d])
                            i_fold += 1

                        experiments_details[experiment][d] = np.mean(fold_scores)

                    print()
                    print('*' * 40)
                    print('experiment {} - demographics: {}, metric: equal odds for {}'.format(experiment, demographic_type, metric))
                    for i in range(len(demo_scores)):
                        for j in range(i + 1, len(demo_scores)):
                            print(demo_scores)
                            print('   equal odds between {} and {}: {}'.format(
                                demo_scores[i], demo_scores[j], np.abs(
                                    experiments_details[experiment][demo_scores[i]] -
                                    experiments_details[experiment][demo_scores[j]]
                                    )
                            ))
                    print('*' * 40)
                    print()

########### Plot performances across folds
    def _get_resultsmetrics(self, resultconfig, metric):
        """Given a result + config dictionary, loads the fold scores for that particular metric

        Args:
            resultconfig (dict): dictionary where result leads to a result file, and config leads to a config file
            metric (_type_): metric we want the data for
        """
        nfolds = resultconfig['config']['ml']['nfolds']['full']
        data_plot = []
        for i in range(nfolds):
            if i in resultconfig['result']:
                if resultconfig['result'][i][metric] == -1:
                    print('undefined {} for fold {}'.fold(metric, i))
                else:
                    data_plot.append(resultconfig['result'][i][metric])
        return data_plot

    def _plot_multiple_metric_boxplots(self, results, metric):
        """Generate the *metric* boxplots across various experiments in results

        Args:
            results (dict): file as generated in the function self._load_data(...) where the key indicates
        the label of the experiment, and the value is a dictionary where the entry at *result* is the result
        file [as generated in the non nested cross validation script], and the entry at *config* is the config file
        from the ml experiment
            metric (str): metric for which to plot the boxplot
        """
        plt.figure(figsize=(14, 6))
        x = self._settings['style']['xmargins']
        xs = []
        xlabels = []
        colours = self._get_colour(len(results))
        
        for i, key in enumerate(results):
            key_plot = self._get_resultsmetrics(results[key], metric)
            self._plot_single_boxplot(key_plot, colours[i], x)
            xs.append(x)
            xlabels.append(key)
            x += self._settings['style']['xspacing']

            if self._settings['print']:
                print('experiment {} for metric {}'.format(key, metric))
                print('    mean: {}\n    std: {}'.format(np.mean(key_plot), np.std(key_plot)))
                print()
        plt.xlim([
            0, 
            x - self._settings['style']['xspacing']+
            self._settings['style']['xmargins']])

        plt.xticks(xs, xlabels)
        plt.ylabel(metric)
        plt.xlabel('subexperiment')

        if self._settings['save']:
            self._savefig('{}_scores_upper'.format(metric))
        if self._settings['show']:
            plt.show()
        else:
            plt.close()


    def _plot_multiple_boxplots(self, results):
        for metric in self._settings['data']['metrics']:
            self._plot_multiple_metric_boxplots(results, metric)

    def test(self, settings):
        results_files, config_files = self._crawl_files()
        results = self._load_data(results_files, config_files)

        if self._settings['boxplot']:
            self._plot_multiple_boxplots(results)

        if self._settings['fairness']:
            self._get_fairness_barplots(results)

        if self._settings['equal_odds']:
            self._get_onemetric_onedemographics_fold_equal_odds(results)


    
