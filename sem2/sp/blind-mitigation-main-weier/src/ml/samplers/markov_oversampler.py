import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter
import json
import markovify
import random
from scipy.stats import gaussian_kde
from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

from ml.samplers.shufflers.shuffler import Shuffler

class MarkovOversampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'markov chain oversampling'
        self._notation = 'mcos'
        self._groupby_keys = ['year']
        # USAGE: change the groupby_keys here. 
        # Choose from [[], ['binconcepts'], ['year'], ['field'], ['language'], ['year', 'language'], ['year','field'], ['field','language']]
        self._rebalancing_mode = self._settings['ml']['oversampler']['rebalancing_mode'] 
        self._shuffler = Shuffler(settings)

    def actn_seq_decode(self, seq):
        one_hot_sequence = np.zeros((len(seq), 4))
        for i, val in enumerate(seq):
            one_hot_sequence[i][val] = 1.0
        return one_hot_sequence.tolist()   
         
    def time_seq_restore_lists(self, input_str):
        lists = [i.split() for i in input_str.split(',')]
        lists = [[int(x) for x in sublist] for sublist in lists]

        origin_lists = []
        for sublist in lists:
            temp = [0.0] * 6
            for index in sublist:
                temp[index] = (len(sublist)-1)*0.1 + np.random.uniform(0, 0.049)
            origin_lists.append(temp)
        return origin_lists

    def sample_length(self, file_path, groupby_keys, values):
        lengths_df = pd.read_csv(file_path)
        length_df = lengths_df[lengths_df['groupby_keys'] == str(groupby_keys)]

        for key, value in values.items():
            length_df = length_df[length_df[key] == value]

        length = length_df['length']

        kde = gaussian_kde(length)
        possible_lengths = np.arange(min(length), max(length)+1)
        probabilities = kde.evaluate(possible_lengths)

        sampled_length = np.random.choice(possible_lengths, p=probabilities/probabilities.sum())

        print("Sampled Length:", sampled_length)
        return sampled_length

    def generate_sequence(self, actn_chain, time_chain, length):
        sequences = []
        num_attempts = 0

        while len(sequences) < length and num_attempts < 200:
            actn_generated = actn_chain.make_sentence(tries=2000)
            time_generated = time_chain.make_sentence(tries=2000)
            if actn_generated is not None and time_generated is not None:
                # Decode the actn_sequence
                actn_sequence = self.actn_seq_decode(list(map(int, actn_generated.split())))
                # Decode the time_sequence
                time_sequence = self.time_seq_restore_lists(time_generated)
                sequence = [actn + time for actn, time in zip(actn_sequence, time_sequence)]
                for seq in sequence:
                    sequences.append(seq)
                #print(len(sequence))  

            num_attempts += 1

        if len(sequences) >= length:
            #print(sequences[:length])
            return sequences[:length]
        else:
            return "Failed to generate sequence after 200 attempts."

    def _oversample(self, sequences:list, labels:list, oversampler:list, demographics,  sampling_strategy:dict, only:str='none') -> Tuple[list, list, list]:
        """Oversamples x based on oversampler, according to the sampling_strategy.

        Args:
            sequences (list): sequences of interaction
            labels (list): target
            oversampler (list): list of the attributes by which to oversample, corresponding to the entries in x
            sampling_strategy (dict): dictionary with the keys as classes, and the values as number of samples to get, or str = 'all' if
            equally balanced
            only: if oversampling one class only, name of the class to retain
        """
        assert len(labels) == len(sequences)
        self._ros = ros(
            random_state = self._settings['seeds']['oversampler'],
            sampling_strategy = sampling_strategy
        )

        indices = [[idx] for idx in range(len(sequences))]
        indices_resampled, _ = self._ros.fit_resample(indices, oversampler)


        #### Part where you code your data augmentation technique
        # Begin block
        # 1) the indices in indices_resampled are the indices of the data you have to augment. 
        # Here, it makes sure that all sequences are at least once in their original shape in the training set
        potential_shuffles = [idx[0] for idx in indices_resampled]
        print(potential_shuffles)
        [potential_shuffles.remove(idx) for idx in range(len(sequences))]
        assert len(potential_shuffles) == (len(indices_resampled) - len(indices))
        # 1) the indices in indices_resampled are the indices of the data you have to augment. 
        # Here, it makes sure that all sequences are at least once in their original shape in the training set
        # End block

        # 2) Objects storing the sequences which you will edit. Here, I called it shuffled because I shuffled the sequences.
        shuffled_sequences = []
        shuffled_oversampler = []
        shuffled_labels = []
        shuffled_indices = []
        # 2) 

        ### Begin EDIT BLOCK
        # 3) Actual part which you can change
        
        groupby_keys = self._groupby_keys
        groupby_keys_str = '_'.join(groupby_keys) if isinstance(groupby_keys, list) else groupby_keys

        for idx in potential_shuffles:
            # print(sequences[0])
            # exit(1)
            if np.random.rand() < 1 / self._settings['ml']['oversampler']['shuffler']['shuffling_coin']:
                #print(groupby_keys_str)
                #print('shuffling')
                length_path = '../data/beerslaw/lengths.csv'
                seq_year = demographics['year'][idx]
                seq_field = demographics['field'][idx]
                seq_language = demographics['language'][idx]
                seq_label = labels[idx]

                if len(groupby_keys) == 0:
                    actn_chain_path = f"../data/beerslaw/markov_chains/all_df_actn.json"
                    time_chain_path = f"../data/beerslaw/markov_chains/all_df_time.json"
                    length = self.sample_length(length_path, 'binconcepts', {})
                if len(groupby_keys) == 1:
                    if groupby_keys[0] == 'binconcepts':  
                        actn_chain_path = f"../data/beerslaw/markov_chains/binconcepts_{seq_label}_actn.json"
                        time_chain_path = f"../data/beerslaw/markov_chains/binconcepts_{seq_label}_time.json"
                        length = self.sample_length(length_path, groupby_keys, {groupby_keys[0]: seq_label})
                    if groupby_keys[0] == 'year':
                        actn_chain_path = f"../data/beerslaw/markov_chains/year_{seq_year}_actn.json"
                        time_chain_path = f"../data/beerslaw/markov_chains/year_{seq_year}_time.json"
                        length = self.sample_length(length_path, groupby_keys, {groupby_keys[0]: seq_year})
                    if groupby_keys[0] == 'field':
                        actn_chain_path = f"../data/beerslaw/markov_chains/field_{seq_field}_actn.json"
                        time_chain_path = f"../data/beerslaw/markov_chains/field_{seq_field}_time.json"
                        length = self.sample_length(length_path, groupby_keys, {groupby_keys[0]: seq_field})
                    if groupby_keys[0] == 'language':
                        actn_chain_path = f"../data/beerslaw/markov_chains/language_{seq_language}_actn.json"
                        time_chain_path = f"../data/beerslaw/markov_chains/languege_{seq_language}_time.json"
                        length = self.sample_length(length_path, groupby_keys, {groupby_keys[0]: seq_language})
                if len(groupby_keys) == 2:
                    if groupby_keys[0] == 'year':
                        if groupby_keys[0] == 'field':
                            actn_chain_path = f"../data/beerslaw/markov_chains/{'_'.join([groupby_keys_str, seq_year, seq_field])}_actn.json"
                            time_chain_path = f"../data/beerslaw/markov_chains/{'_'.join([groupby_keys_str, seq_year, seq_field])}_time.json"
                            length = self.sample_length(length_path, groupby_keys, {groupby_keys[0]: seq_year, groupby_keys[1]: seq_field})
                        if groupby_keys[0] == 'language':
                            actn_chain_path = f"../data/beerslaw/markov_chains/{'_'.join([groupby_keys_str, seq_year, seq_language])}_actn.json"
                            time_chain_path = f"../data/beerslaw/markov_chains/{'_'.join([groupby_keys_str, seq_year, seq_language])}_time.json"
                            length = self.sample_length(length_path, groupby_keys, {groupby_keys[0]: seq_year, groupby_keys[1]: seq_language})                          
                    if groupby_keys[0] == 'field':
                        actn_chain_path = f"../data/beerslaw/markov_chains/{'_'.join([groupby_keys_str, seq_field, seq_language])}_actn.json"
                        time_chain_path = f"../data/beerslaw/markov_chains/{'_'.join([groupby_keys_str, seq_field, seq_language])}_time.json"
                        length = self.sample_length(length_path, groupby_keys, {groupby_keys[0]: seq_field, groupby_keys[1]: seq_language})
                
                with open(actn_chain_path, 'r') as f:
                    actn_chain_json = json.load(f)
                actn_chain =  markovify.Text.from_json(actn_chain_json)

                with open(time_chain_path, 'r') as f:
                    time_chain_json = json.load(f)
                time_chain =  markovify.Text.from_json(time_chain_json)

                #length = random.randint(25, 80)
                markov_sequence = (self.generate_sequence(actn_chain, time_chain, length))
                shuffled_sequences.append(markov_sequence)
                
                #print(type(self._shuffler.shuffle(sequences[idx])))
                #print((self._shuffler.shuffle(sequences[idx])))
                
            else:
                # print('not shuffling')
                shuffled_sequences.append(sequences[idx])
        # 3) Actual part which you can implement for your data augmentation
        ### End EDIT BLOCK

            # Saving the data
            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

        # Adding the original sequences
        [shuffled_sequences.append(sequences[idx]) for idx in range(len(sequences))]
        [shuffled_labels.append(labels[idx]) for idx in range(len(labels))]
        [shuffled_indices.append(idx) for idx in range(len(labels))]
        [shuffled_oversampler.append(oversampler[idx]) for idx in range(len(oversampler))]


        print('distrbution os after the sampling: {}'.format(sorted(Counter(shuffled_oversampler).items())))
        print('labels after sampling: {}'.format(Counter(shuffled_labels)))
        return shuffled_sequences, shuffled_labels, shuffled_indices     


    def _equal_oversampling(self, sequences:list, oversampler:list, labels:list, demographics:dict) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Rebalances all classes equally

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        return self._oversample(sequences, labels, oversampler, demographics, 'all')

    def sample(self, sequences:list, oversampler:list, labels:list, demographics:list) -> Tuple[list, list]:
        """Chooses the mode of oversampling

        1. equal oversampling: All instances are oversampled by n, determined by imbalanced-learn
        2. Major oversampling: Only the largest class is oversampled
        3. Only Major Oversampling: Only the largest class is oversampled, all other classes are taken out the training set
        4. Minor oversampling: Only the smallest class is oversampled
        5. Only Minor Oversampling: Only the smallest class is oversampled, all other classes are taken out the training set

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        if self._settings['ml']['oversampler']['rebalancing_mode'] == 'equal_balancing':
            return self._equal_oversampling(sequences, oversampler, labels, demographics)
        
    def get_indices(self) -> np.array:
        return self._indices