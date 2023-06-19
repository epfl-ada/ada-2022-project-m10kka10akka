import random
import numpy as np

class Shuffler:

    def __init__(self, settings:dict):
        self._name = 'shuffler'
        self._settings = dict(settings)

        random.seed(self._settings['seeds']['shuffler'])
        self._select_shuffling()

    def _select_shuffling(self):
        if self._settings['ml']['oversampler']['shuffler']['mode'] == 'simple':
            self._shuffle = self._simple_shuffle
        if self._settings['ml']['oversampler']['shuffler']['mode'] == 'chunk':
            self._chunk_size = self._settings['ml']['oversampler']['shuffler']['chunk_size']
            self._shuffle = self._chunk_shuffle

    def _add_noise(self, sub_sequence:list) -> list:
        """ Adds random noise within ±10% of the original value to the non-zero elements of the sequence

        Args:
            sub_sequence (list): The sublist of the sequence to which noise is to be added

        Returns:
            list: The sublist with added noise
        """
        noise_range = 0.1  # ±10% range for noise
        for i, value in enumerate(sub_sequence):
            if value != 0:
                noise = np.random.uniform(-value*noise_range, value*noise_range)
                sub_sequence[i] += noise
        return sub_sequence
    
    def _simple_shuffle(self, sequence:list) -> list:
        """ Creates a synthetic instance of a timeserie based on an existing time serie in the following way:
        1. Select an index at random between the beginning of the ts and its end
        2. This index is the breaking point. Break the sequence at that index
        3. Shuffle the beginning of the sequence and the end of the sequence (beginning and end being delimited 
        by the breaking point)

        Args:
            sequence (list): sequence to shuffle

        Returns:
            list: synthetic instance based on the sequence
        """
        breaking_point = np.random.randint(1, len(sequence))
        '''
        for i in range(len(sequence)):
            sequence[i][-6:] = self._add_noise(sequence[i][-6:])
        '''

        return [*sequence[breaking_point:], *sequence[:breaking_point]]

    def _chunk_shuffle(self, sequence:list) -> list:
        """ Creates a synthetic instance of a timeserie based on an existing time serie in the following way:
        1. create chunks of self._chunk_size timesteps
        2. Shuffle those chunks around to create the new sequence
        If the sequence size is smaller than self._chunk_size -> proceed to self._simple_shuffle
        

        Args:
            sequence (list): sequence to shuffle

        Returns:
            list: synthetic instance based on the sequence
        """
        if len(sequence) - 1 < 2 * self._chunk_size:
            return self._simple_shuffle(sequence)

        chunks_breaking_points = range(self._chunk_size, len(sequence), self._chunk_size)
        chunks = []
        curr_breaking_point = 0
        for bp in chunks_breaking_points[:-1]:
            chunks.append([sequence[i] for i in range(curr_breaking_point, bp)])
            curr_breaking_point = bp
        chunks.append([s for s in sequence[bp:]])
        random.shuffle(chunks)
        new_sequence = []
        for chunk in chunks:
            new_sequence = new_sequence + chunk

        return new_sequence

        

    def shuffle(self, sequence):
        return self._shuffle(sequence)