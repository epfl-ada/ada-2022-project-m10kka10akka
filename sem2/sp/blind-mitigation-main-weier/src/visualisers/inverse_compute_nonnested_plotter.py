import os
import re
from typing import Tuple
from unittest import result
import yaml
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ml.scorers.binary_scorer import BinaryClfScorer
from visualisers.plotter import Plotter

class InverseNonNestedPlotter(Plotter):

    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'inversecompute non nested plotter'
        self._notation = 'icnnpltr'

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
            '#594DD4', '#213AF2', '#A8BCC7', '#618595', '#618595'
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
        # configs_files.sort()
        # results_files.sort()
        # for i in range(max(len(configs_files), len(results_files))):
        #     print(configs_files[i])
        #     print(results_files[i])
        #     print()
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

    def _inverse_simple(self, ys:list) -> list:
        return [np.abs(y-1) for y in ys]
    
    def _inverse_probs(self, yprobs:list) -> list:
        return [[yp[1], yp[0]] for yp in yprobs]

    def _inverse_ys(self, ytrues:list, ypreds:list, yprobs:list) -> Tuple[list, list, list]:

        return self._inverse_simple(ytrues), self._inverse_simple(ypreds), self._inverse_probs(yprobs)


###### CSV functions
    def _get_csv_data(self, resultsconfig:dict):
        csv_data = {}
        for experiment in resultsconfig:
            configs = dict(resultsconfig[experiment]['config'])

            key = '{}{}'.format(configs['experiment']['root_name'], configs['experiment']['name'])

            oversampling_mode = configs['ml']['pipeline']['oversampler']
            if oversampling_mode == 'none':
                oversampling_proportion = 'none'
                oversampling_attribute = 'none'

            else:
                if oversampling_mode == 'shuffle_ros':
                    oversampling_mode += '_{}'.format(configs['ml']['oversampler']['shuffler']['mode'])
                    if oversampling_mode == 'chunk':
                        oversampling_mode += '_{}'.format(configs['ml']['oversampler']['shuffler']['chunk_size'])

                oversampling_proportion = configs['ml']['oversampler']['rebalancing_mode']

                oversampling_attribute = '_'.join(configs['ml']['oversampler']['oversampling_col'])


            experiment_data = {
                'key': key,
                'oversampling_model': oversampling_mode,
                'oversampling_proportion': oversampling_proportion,
                'oversampling_attribute': oversampling_attribute
            }
            metrics = self._settings['data']['fairness_metrics'] + self._settings['data']['metrics']
            csv_data[experiment] = {metric: dict(experiment_data) for metric in metrics}

        return csv_data
        
########### Get Fairness Metrics
    def _get_overall_scorer(self, ytrue:list):
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
        else:
            print('not implemented for multi class yet')
            raise NotImplementedError

    def _get_fold_scorer(self, n_classes: int, metrics:list):
        """selects the appropriate scorer + looks into the correct metrics

        Args:
            n_classes (int): number of classes in the experiment
            metrics (list): metrics for which to compute scores
        """
        new_settings = dict(self._settings)
        new_settings['ml'] = {
            'scorer': {'scoring_metrics' : metrics}
        }
        if n_classes == 2:
            return BinaryClfScorer(new_settings)
        else:
            print('not implemented for multi class yet')
            raise NotImplementedError

    
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
            fold = 0
            results = dict(resultsconfig[experiment]['result'])
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
            
            y_trues[experiment] = self._inverse_simple(y_true)
            y_preds[experiment] = self._inverse_simple(y_pred)
            y_probs[experiment] = self._inverse_probs(y_prob)
            demogs[experiment] = {k: v for k, v in demo_list.items()}

        return y_trues, y_preds, y_probs, demogs

    def _get_onemetric_onedemographics_overall_boxplots(self, ytrues:list, ypreds:list, yprobs:list, demographics:dict):
        scorer = self._get_overall_scorer(ytrues[list(ytrues.keys())[0]])
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
                if self._settings['save']:
                    self._savefig('{}_{}_overall_barplot'.format(metric, demographic_type), 'fairness_metrics')
                if self._settings['show']:
                    plt.show()
                else:
                    plt.close()

    def _get_onemetric_onedemographics_overall_csv(self, ytrues:list, ypreds:list, yprobs:list, demographics:list, csv_data:dict):
        
        experiment_table = dict(csv_data)
        for demographic_type in self._settings['data']['demographics']:
            for metric in self._settings['data']['fairness_metrics']:
                # demo_attributes = np.unique(demographics[list(ytrues.keys())[0]][demographic_type])
                for i_e, experiment in enumerate(ytrues):
                    ytrue = ytrues[experiment]
                    ypred = ypreds[experiment]
                    yprob = yprobs[experiment]
                    demo = demographics[experiment]
                    demo_attributes = demographics[experiment][demographic_type]

                    ytrue, ypred, yprob = self._inverse_ys(ytrue, ypred, yprob)
                    scorer = self._get_overall_scorer(ytrue)

                    scores = scorer.get_fairness_scores(ytrue, ypred, yprob, demo[demographic_type], [metric])
                
                    for d in demo_attributes:
                        if scores[metric][d] == -1:
                            print('undefined {} for demographic {} for all folds {}'.format(metric, d))
                            continue
                        experiment_table[experiment][metric]['overall'] = scores[metric][d]

        self._update_csv(experiment_table)

    def _get_onemetric_onedemographics_fold_boxplots(self, resultsconfig:dict):
        for demographic_type in self._settings['data']['demographics']:
            for metric in self._settings['data']['fairness_metrics']:
                plt.figure(figsize=(self._figwidth, self._figheight))
                plt.title('{} for {}'.format(metric, demographic_type))
                e = 0

                experiments_details = {}
                for i_e, experiment in enumerate(resultsconfig):
                    results = resultsconfig[experiment]['result']
                    scorer = self._get_fold_scorer(
                        resultsconfig[experiment]['config']['experiment']['n_classes'], 
                        self._settings['data']['fairness_metrics']
                    )
                    colours = self._get_colour(len(results))
                    demo_scores = np.unique(results['demographics'][demographic_type])

                    experiments_details[experiment] = {}
                    x = e
                    
                    xs = []
                    ys = []
                    labels = []
                    i_fold = 0
                    for d in demo_scores:
                        i_fold = 0
                        fold_scores = []
                        while (i_fold in results):
                            test_demos = [results['demographics'][demographic_type][idx] for idx in results[i_fold]['test_index']]

                            if d not in test_demos:
                                i_fold += 1
                                continue
                        
                            demo_indices = [idx for idx in range(len(test_demos)) if test_demos[idx] == d]
                            d_indices = [results[i_fold]['test_index'][didx] for didx in demo_indices]
                            d_demos = [test_demos[didx] for didx in demo_indices]
                            d_trues = [results['y'][didx] for didx in d_indices]
                            d_pbs = [results[i_fold]['y_proba'][didx] for didx in demo_indices]
                            d_pds = [results[i_fold]['y_pred'][didx] for didx in demo_indices]

                            d_trues, d_pds, d_pbs = self._inverse_ys(d_trues, d_pds, d_pbs)
                            # debug
                            debug = [td for td in test_demos if td==d]
                            assert len(debug) == len(d_demos) # debug to say that indexing worked correctly 

                            d_score = scorer.get_fairness_scores(d_trues, d_pds, d_pbs, d_demos, [metric])
                            if d_score[metric][d] == -1:
                                i_fold += 1
                                print('undefined metric for fold {}'.format(i_fold - 1))
                                continue

                            fold_scores.append(d_score[metric][d])
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
                plt.xticks(xs, demo_scores)
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

    def _get_onemetric_onedemographics_fold_csv(self, resultsconfig:dict, csv_data:dict):
        experiment_table = dict(csv_data)
        for demographic_type in self._settings['data']['demographics']:
            for metric in self._settings['data']['fairness_metrics']:

                for i_e, experiment in enumerate(resultsconfig):
                    results = resultsconfig[experiment]['result']
                    scorer = self._get_fold_scorer(
                        resultsconfig[experiment]['config']['experiment']['n_classes'], 
                        self._settings['data']['fairness_metrics']
                    )
                    demo_scores = np.unique(results['demographics'][demographic_type])

                    i_fold = 0
                    for d in demo_scores:
                        i_fold = 0
                        fold_scores = []
                        while (i_fold in results):
                            test_demos = [results['demographics'][demographic_type][idx] for idx in results[i_fold]['test_index']]

                            if d not in test_demos:
                                i_fold += 1
                                continue

                            demo_indices = [idx for idx in range(len(test_demos)) if test_demos[idx] == d]
                            d_indices = [results[i_fold]['test_index'][didx] for didx in demo_indices]
                            d_trues = [results['y'][didx] for didx in d_indices]
                            d_pbs = [results[i_fold]['y_proba'][didx] for didx in demo_indices]
                            d_pds = [results[i_fold]['y_pred'][didx] for didx in demo_indices]

                            d_trues, d_pds, d_pbs = self._inverse_ys(d_trues, d_pds, d_pbs)
                        
                            d_score = scorer.get_scores(d_trues, d_pds, d_pbs)
                            if d_score[metric] == -1:
                                i_fold += 1
                                print('undefined metric for fold {}'.format(i_fold - 1))
                                continue

                            fold_scores.append(d_score[metric])
                            i_fold += 1

                        experiment_table[experiment][metric]['{}_{}-mean'.format(demographic_type, d)] = np.mean(fold_scores)
                        experiment_table[experiment][metric]['{}_{}-std'.format(demographic_type, d)] = np.std(fold_scores)

        self._update_csv(experiment_table)

    def _get_combineddemo_barplot(self, resultsconfig:dict):
        for experiment in resultsconfig:
            scorer = self._get_fold_scorer(
                        resultsconfig[experiment]['config']['experiment']['n_classes'], 
                        self._settings['data']['metrics']
            )
            results = resultsconfig[experiment]['result']
            recombined_demos = ['' for _ in range(len(results['y']))]
            for demographics in self._settings['data']['combined_demographics']:
                recombined_demos = [
                    '{}_{}'.format(recombined_demos[idx], results['demographics'][demographics][idx]) 
                    for idx in range(len(recombined_demos))
                    ]
            recombined_demos = [rd[1:] for rd in recombined_demos]

            fold = 0
            demos = []
            ys = []
            ypreds = []
            yprobs = []
            while fold in results and fold < resultsconfig[experiment]['config']['ml']['nfolds']['full']:
                ys = ys + [results['y'][tidx] for tidx in results[fold]['test_index']]
                ypreds = ypreds + [yp for yp in results[fold]['y_pred']]
                yprobs = yprobs + [ypp for ypp in results[fold]['y_proba']]
                demos = demos + [recombined_demos[tidx] for tidx in results[fold]['test_index']]

                ys, ypreds, yprobs = self._inverse_ys(ys, ypreds, yprobs)
                fold += 1

            for metric in self._settings['data']['metrics']:
                plt.figure(figsize=(self._figwidth, self._figheight))
                plt.title('{} for {}'.format(metric, ' '.join(self._settings['data']['combined_demographics'])))

                colours = self._get_colour(len(np.unique(recombined_demos)))
                x = 0
                xs = []
                heights = []
                rds = []
                ticks = []
                for rd in np.unique(recombined_demos):
                    rds.append(rd)


                    rds_indices = [idx for idx in range(len(demos)) if demos[idx] == rd]
                    yt = [ys[idx] for idx in rds_indices]
                    ypds = [ypreds[idx] for idx in rds_indices]
                    ypbs = [yprobs[idx] for idx in rds_indices]

                    d_score = scorer.get_scores(yt, ypds, ypbs)
                    if d_score[metric] == -1:
                        print('undefined metric {}'.format(metric))
                        continue
                    
                    xs.append(x)
                    x += self._settings['style']['xspacing']
                    heights.append(d_score[metric])
                    ticks.append(rd)

                
                exclude_re = re.compile('({})'.format('|'.join(self._settings['data']['exclude_demographics'])))
                indices = [i for i in range(len(ticks)) if len(exclude_re.findall(ticks[i])) == 0]
                xs = [xx for xx in xs[:len(indices)]]
                heights = [heights[idx] for idx in indices]
                ticks = [ticks[idx] for idx in indices]

                plt.bar(x=xs, height=heights, width=self._settings['style']['bar_width'], color=colours)
                plt.xticks(xs, ticks, rotation=self._settings['style']['rotation'])
                plt.ylim([0, 1])
                plt.title('{} for combined demographics {} - {}'.format(metric, ' '.join(self._settings['data']['combined_demographics']), experiment))

                plt.legend()
                if self._settings['save']:
                    self._savefig('{}_{}_combineddemographics_overall_barplot'.format(metric, ' '.join(self._settings['data']['combined_demographics'])), 'fairness_metrics')
                if self._settings['show']:
                    plt.show()
                else:
                    plt.close()

    def _get_combineddemo_barplot_print(self, resultsconfig:dict, csv_results:dict):
        print(resultsconfig.keys())
        experiment_table = dict(csv_results)
        for experiment in resultsconfig:
            scorer = self._get_fold_scorer(
                        resultsconfig[experiment]['config']['experiment']['n_classes'], 
                        self._settings['data']['metrics']
            )
            results = resultsconfig[experiment]['result']
            recombined_demos = ['' for _ in range(len(results['y']))]
            for demographics in self._settings['data']['combined_demographics']:
                recombined_demos = [
                    '{}_{}'.format(recombined_demos[idx], results['demographics'][demographics][idx]) 
                    for idx in range(len(recombined_demos))
                    ]
            recombined_demos = [rd[1:] for rd in recombined_demos]


            fold = 0
            demos = []
            ys = []
            ypreds = []
            yprobs = []
            while fold in results and fold < resultsconfig[experiment]['config']['ml']['nfolds']['full']:
                ys = ys + [results['y'][tidx] for tidx in results[fold]['test_index']]
                ypreds = ypreds + [yp for yp in results[fold]['y_pred']]
                yprobs = yprobs + [ypp for ypp in results[fold]['y_proba']]
                demos = demos + [recombined_demos[tidx] for tidx in results[fold]['test_index']]
                fold += 1

            for metric in self._settings['data']['metrics']:
                for rd in np.unique(recombined_demos):
                    

                    rds_indices = [idx for idx in range(len(demos)) if demos[idx] == rd]
                    yt = [ys[idx] for idx in rds_indices]
                    ypds = [ypreds[idx] for idx in rds_indices]
                    ypbs = [yprobs[idx] for idx in rds_indices]

                    yt, ypds, ypbs = self._inverse_ys(yt, ypds, ypbs)
                    d_score = scorer.get_scores(yt, ypds, ypbs)

                    experiment_table[experiment][metric]['combinedoverall_{}_mean'.format(rd)] = d_score[metric]

        self._update_csv(experiment_table)

    def _pairwise_distance_boxplots(self, resultconfigs:dict):
        for experiment in resultconfigs:
            results = resultconfigs[experiment]['result']
            config = resultconfigs[experiment]['config']
            for demographic in self._settings['data']['demographics']:
                fold = 0
                demos = []
                ys = []
                ypreds = []
                yprobs = []
                while fold in results and fold < config['ml']['nfolds']['full']:
                    ys = ys + [results['y'][tidx] for tidx in results[fold]['test_index']]
                    ypreds = ypreds + [yp for yp in results[fold]['y_pred']]
                    yprobs = yprobs + [ypp for ypp in results[fold]['y_proba']]
                    demos = demos + [results['demographics'][demographic][tidx] for tidx in results[fold]['test_index']]
                    fold += 1

                for metric in self._settings['data']['metrics']:
                    scorer = self._get_fold_scorer(
                            resultconfigs[experiment]['config']['experiment']['n_classes'], 
                            self._settings['data']['metrics']
                    )

                    unique_demos = np.unique(demos)
                    distance_table = {d : {dd : -1 for dd in unique_demos} for d in unique_demos}
                    score_table = {}
                    for d_attribute in unique_demos:
                        exclude_re = re.compile('({})'.format('|'.join(self._settings['data']['exclude_demographics'])))
                        if len(exclude_re.findall(d_attribute)) > 0:
                            continue

                        demos_indices = [idx for idx in range(len(demos)) if demos[idx] == d_attribute]
                        demos_preds = [ypreds[idx] for idx in demos_indices]
                        demos_probs = [yprobs[idx] for idx in demos_indices]
                        demos_trues = [ys[idx] for idx in demos_indices]

                        demos_trues, demos_preds, demos_probs = self._inverse_ys(demos_trues, demos_preds, demos_probs)

                        d_scores = scorer.get_scores(demos_trues, demos_preds, demos_probs)
                        score_table[d_attribute] = d_scores[metric]
                    

                    for d_i in score_table:
                        for d_j in score_table:
                            distance_table[d_i][d_j] = np.abs(score_table[d_i] - score_table[d_j])

                    x = 0
                    xs = []
                    xlabels = []
                    colours = self._get_colour(len(unique_demos))
                    for d_i, d in enumerate(unique_demos):
                        d_scores = [distance_table[d][dd] for dd in distance_table[d] if d != dd]
                        xs.append(x)
                        xlabels.append(d)
                        self._plot_single_boxplot(d_scores, colours[d_i], x)
                        x += self._settings['style']['xspacing']

                    plt.xlim([
                        - 
                        self._settings['style']['xmargins'], 
                        x - self._settings['style']['xspacing']+
                        self._settings['style']['xmargins']])

                    plt.xticks(xs, xlabels)
                    plt.ylabel(metric)
                    plt.ylim([0, 1])
                    plt.xlabel('demographic')

                    if self._settings['save']:
                        self._savefig('{}_{}_distanceplot'.format(metric, demographic))
                    if self._settings['show']:
                        plt.show()
                    else:
                        plt.close()

    def _get_fairness_results(self, resultconfigs:dict):
        if self._settings['overall']:
            if self._settings['combined'] and self._settings['barplot']:
                
                if self._settings['dump']:
                    csv_data = self._get_csv_data(resultconfigs)
                    self._get_combineddemo_barplot_print(resultconfigs, csv_data)
                else:
                    self._get_combineddemo_barplot(resultconfigs)
            if self._settings['pairwise']:
                self._pairwise_distance_boxplots(resultconfigs)
            else:
                if self._settings['show'] or self._settings['save']:
                    y_trues, y_preds, y_probs, demographics = self._get_fairness_data(resultconfigs)
                    self._get_onemetric_onedemographics_overall_boxplots(y_trues, y_preds, y_probs, demographics)
                if self._settings['print'] or self._settings['dump']:
                    y_trues, y_preds, y_probs, demographics = self._get_fairness_data(resultconfigs)
                    csv_data = self._get_csv_data(resultconfigs)
                    self._get_onemetric_onedemographics_overall_csv(y_trues, y_preds, y_probs, demographics, csv_data)
        elif self._settings['fold']:
            if self._settings['show'] or self._settings['save']:
                self._get_onemetric_onedemographics_fold_boxplots(resultconfigs)
            if self._settings['print'] or self._settings['dump']:
                csv_data = self._get_csv_data(resultconfigs)
                self._get_onemetric_onedemographics_fold_csv(resultconfigs, csv_data)

#### Compute equal odds
    def _get_onemetric_onedemographics_fold_equal_odds_print(self, resultsconfig:dict, csv_results:dict):
        csv_table = dict(csv_results)
        for metric in self._settings['data']['fairness_metrics']:
            for demographic_type in self._settings['data']['demographics']:
                experiments_details = {}
                for i_e, experiment in enumerate(resultsconfig):
                    scorer = self._get_fold_scorer(
                        resultsconfig[experiment]['config']['experiment']['n_classes'], 
                        self._settings['data']['metrics']
                    )
                    results = resultsconfig[experiment]['result']
                    experiments_details[experiment] = {}
                    i_fold = 0
                    demo_scores = np.unique(results['demographics'][demographic_type])
                    for d in demo_scores:
                        i_fold = 0
                        fold_scores = []
                        while (i_fold in results):
                            test_demos = [results['demographics'][demographic_type][idx] for idx in results[i_fold]['test_index']]

                            if d not in test_demos:
                                i_fold += 1
                                continue

                            demo_indices = [idx for idx in range(len(test_demos)) if test_demos[idx] == d]
                            d_indices = [results[i_fold]['test_index'][didx] for didx in demo_indices]
                            d_trues = [results['y'][didx] for didx in d_indices]
                            d_pbs = [results[i_fold]['y_proba'][didx] for didx in demo_indices]
                            d_pds = [results[i_fold]['y_pred'][didx] for didx in demo_indices]
                            d_trues, d_pds, d_pbs = self._inverse_ys(d_trues, d_pds, d_pbs)

                            d_score = scorer.get_scores(d_trues, d_pds, d_pbs)

                            if d_score[metric] == -1:
                                print('undefined {} for {} at fold {}'.format(metric, d, i_fold))
                                i_fold += 1
                                continue
                            fold_scores.append(d_score[metric])
                            i_fold += 1

                        experiments_details[experiment][d] = np.mean(fold_scores)

                    if self._settings['print']:
                        print()
                        print('*' * 40)
                        print('experiment {} - demographics: {}, metric: equal odds for {}'.format(experiment, demographic_type, metric))
                    for i in range(len(demo_scores)):
                        for j in range(i + 1, len(demo_scores)):
                            if self._settings['print']:
                                print('   equal odds between {} and {}: {}'.format(
                                    demo_scores[i], demo_scores[j], np.abs(
                                        experiments_details[experiment][demo_scores[i]] -
                                        experiments_details[experiment][demo_scores[j]]
                                        )
                                ))
                            csv_table[experiment][metric]['{}_equalodds_{}-{}'.format(demographic_type, demo_scores[i], demo_scores[j])]  = np.abs(
                                        experiments_details[experiment][demo_scores[i]] -
                                        experiments_details[experiment][demo_scores[j]]
                                        )
                    
                    if self._settings['print']:
                        print('*' * 40)
                        print()

        self._update_csv(csv_table)

########### Plot performances across folds
    def _get_resultsmetrics(self, resultconfig, metric):
        """Given a result + config dictionary, loads the fold scores for that particular metric

        Args:
            resultconfig (dict): dictionary where result leads to a result file, and config leads to a config file
            metric (_type_): metric we want the data for
        """
        
        nfolds = resultconfig['config']['ml']['nfolds']['full']
        scorer = self._get_fold_scorer(
                        resultconfig['config']['experiment']['n_classes'], 
                        self._settings['data']['metrics']
        )
        results = dict(resultconfig['result'])

        data_plot = []
        for i_fold in range(nfolds):
            if i_fold in results:
                y_trues = [results['y'][tidx] for tidx in results[i_fold]['test_index']]
                y_pbs = [yb for yb in results[i_fold]['y_proba']]
                y_pds = [yd for yd in results[i_fold]['y_pred']]

                y_trues, y_pds, y_pbs = self._inverse_ys(y_trues, y_pds, y_pbs)
                
                fold_scores = scorer.get_scores(y_trues, y_pds, y_pbs)
                if fold_scores[metric] == -1:
                    print('undefined {} for fold {}'.fold(metric, i_fold))
                else:
                    data_plot.append(fold_scores[metric])
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

    def _plot_multiple_metric_csv(self, results, metric, csv_results: dict):
        """Generate the *metric* boxplots across various experiments in results

        Args:
            results (dict): file as generated in the function self._load_data(...) where the key indicates
        the label of the experiment, and the value is a dictionary where the entry at *result* is the result
        file [as generated in the non nested cross validation script], and the entry at *config* is the config file
        from the ml experiment
            metric (str): metric for which to plot the boxplot
        """
        experiment_table = dict(csv_results)
        for i, key in enumerate(results):
            scores = self._get_resultsmetrics(results[key], metric)

            if self._settings['print']:
                print('experiment {} for metric {}'.format(key, metric))
                print('    mean: {}\n    std: {}'.format(np.mean(scores), np.std(scores)))
                print()

            experiment_table[key][metric]['overall_mean'] = np.mean(scores)
            experiment_table[key][metric]['overall_std'] = np.std(scores)

        self._update_csv(experiment_table)

    def _plot_multiple_boxplots(self, resultconfigs):
        for metric in self._settings['data']['metrics']:
            if self._settings['show'] or self._settings['save']:
                self._plot_multiple_metric_boxplots(resultconfigs, metric)
            else:
                csv_data = self._get_csv_data(resultconfigs)
                self._plot_multiple_metric_csv(resultconfigs, metric, csv_data)

    def test(self, settings):
        results_files, config_files = self._crawl_files()
        resultconfigs = self._load_data(results_files, config_files)

        if self._settings['boxplot']:
            self._plot_multiple_boxplots(resultconfigs)

        if self._settings['fairness']:
            self._get_fairness_results(resultconfigs)

        if self._settings['equal_odds']:
            csv_data = self._get_csv_data(resultconfigs)
            self._get_onemetric_onedemographics_fold_equal_odds_print(resultconfigs, csv_data)


    
