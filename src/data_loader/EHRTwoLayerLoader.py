"""
Function: from disk import data files: X (patients) and Y (tags) for sequence classification
this version doesn't apply tf dataset with iterator, but directly padding for all batches
Author: Xiang.Yang
"""
import sys

sys.path.append('./')
import tensorflow as tf
import logging
import numpy as np
import random
import pickle as pkl
from utils.utils import save_dict_to_json
from utils.vocab import load_vocab
from utils.config import get_config_from_json
from utils.utils import get_args

tf.set_random_seed(1234)  # let each random shuffle fixed


class EHRTwoLayerLoader:
    def __init__(self, config, data_path, word_vocab, position_vocab, tag_vocab, training_flag=False):
        self.config = config
        self.data_path = data_path
        self.word_vocab = word_vocab
        self.position_vocab = position_vocab
        self.tag_vocab = tag_vocab
        self.id_pad_word = self.word_vocab['</s>']
        self.id_pad_position = self.position_vocab[0]
        self.id_pad_tag = self.tag_vocab[0]  # not used
        self.training_flag = training_flag
        #print("training flag: {}".format(training_flag))

    def load_text_from_file_for_twolayers(self):
        """
        store into two layer format so that the bottom layer can be firstly embeded by attention, and the second layer
        rnn
        :return:
        """
        try:
            loaded_tags, loaded_sentences, loaded_positions = pkl.load(open(self.data_path, 'rb'))
        except:
            print('File not exists error')
            return
        # one layer of codes as words and correponding days as positions
        tags, patients_visitis, patients_days = [], [], []
        max_time = -1
        # traverse each patient
        for i in range(len(loaded_tags)):
            tags.append(loaded_tags[i])
            patient_visits, patient_days = [], []
            # a patient's information
            loaded_patient_visits, loaded_patient_days = loaded_sentences[i], loaded_positions[i]
            # determine if the data lengths are consistent
            if len(loaded_patient_visits) != len(loaded_patient_days):
                print(len(loaded_patient_visits), len(loaded_patient_days))
                input()
            for j in range(len(loaded_patient_visits)):
                visit_codes, visit_time = loaded_patient_visits[j], loaded_patient_days[j]
                # for one visit
                patient_visits.append(visit_codes)
                # each visit only needs one time(position), 0: days to the first, 1: to the previous, or 2: to the prediction date
                time = visit_time[2]
                # in case error in time
                if time > max_time:
                    max_time = time
                patient_days.append(time)
            
            #print(patient_visits)
            #print(patient_days)
            #input()
            patients_visitis.append(patient_visits)
            patients_days.append(patient_days)
        
        print('max time', max_time)
        return tags, patients_visitis, patients_days

    def input_fn(self, patients_visits, patients_days, tags):
        """
        generate an iterator to fetch batch data
        Args:
            mode: train or valid or test
            sentences: sentences in indices
            tags: tags in indices
            id_pad_word:
            id_pad_tag:

        Returns:

        """
               
        patients_visits = [[[self.word_vocab[word] if word in self.word_vocab else self.id_pad_word 
                            for word in words]
                            for words in patient_visits] 
                            for patient_visits in patients_visits]

        patients_days = [[self.position_vocab[pos] if pos in self.position_vocab else self.id_pad_position 
                            for pos in patient_days]
                            for patient_days in patients_days]

        # the number of visits for each patient
        patients_lengths = [len(patient) for patient in patients_days]
        max_patient_len = np.max(patients_lengths)
        tags = [self.tag_vocab[int(tag)] for tag in tags]

        # the number of codes for each visit
        visits_lengths = []
        for patient in patients_visits:
            visits_lengths.extend([len(words) for words in patient])
        max_visit_len = np.max(visits_lengths)
        # print(max_patient_len)
        print('max patient len: {}, max visit len: {}'.format(max_patient_len, max_visit_len))

        self.num_of_cases = np.sum([1 if i == 1 else 0 for i in tags])
        self.num_of_controls = np.sum([1 if i == 0 else 0 for i in tags])

        print("Positive by negative ratio is {}:{}".format(self.num_of_cases, self.num_of_controls))

        self.datasize = len(tags)
        print("Loaded datasize: {}".format(self.datasize))
        self.max_patient_len = max_patient_len
        self.max_visit_len = max_visit_len

        dataset = list(zip(tags, patients_visits, patients_days, patients_lengths))

        return dataset

    def balance_data(self):
        """
        balance the training set by oversampling, if the imbalance data affect the result, try it
        :return:
        """
        tags, _, _, _ = zip(*self.dataset)
        case_indices, control_indices = [], []
        for i in range(self.datasize):
            if tags[i] == 1:
                case_indices.append(i)
            else:
                control_indices.append(i)
        times = int((self.num_of_controls / self.num_of_cases) * 0.5)
        augmented_cases = []
        for i in range(times):
            cases = [self.dataset[k] for k in case_indices]
            if self.training_flag:
                random.shuffle(cases)
            augmented_cases.extend(cases)
        self.dataset.extend(augmented_cases)

        self.datasize = len(self.dataset)
        print("new dataset size : {}".format(self.datasize))

    def pad_data(self, max_patient_len, max_visit_len):
        # padding
        def padding_codes(patient, org_len, max_patient_len, max_visit_len, pad_id):
            for i in range(len(patient)):
                patient[i].extend([pad_id]*(max_visit_len - len(patient[i])))
            for _ in range(max_patient_len - org_len):
                patient.append([pad_id] * max_visit_len)
            return patient

        def padding_day(patient, org_len, max_patient_len, pad_id):
            for _ in range(max_patient_len - org_len):
                patient.append(pad_id)
            return patient

        # for two layer patient information, only pad the patient-level for rnn, pad one [</s>]
        tags, patients_visits, patients_days, patients_lengths = zip(*self.dataset)
        padded_patients_visits = [
            padding_codes(patient_visits, len(patient_visits), max_patient_len, max_visit_len, self.id_pad_word)
            for patient_visits in patients_visits]
        padded_patients_days = [
            padding_day(patient_days, len(patient_days), max_patient_len, self.id_pad_position)
            for patient_days in patients_days]

        # shuffle
        dataset = list(zip(tags, padded_patients_visits, padded_patients_days, patients_lengths))
        if self.training_flag:
            random.shuffle(dataset)
        self.dataset = dataset

    def next_batch(self, prev_idx):
        b = self.config.batch_size
        upper_bound = np.min([b*(prev_idx+1), self.datasize])
        yield self.dataset[b * prev_idx : upper_bound]

    def get_datasize(self):
        return self.datasize

    def get_dataset(self):
        return self.dataset

    def load_data(self):
        """
        for sequence labeling:
            one line per sentence in X, one line per tag for sentence in Y
        :return:
        """
        # load raw sentences and tags
        tags, sentences, positionss = self.load_text_from_file_for_twolayers()

        dataset = self.input_fn(sentences, positionss, tags)
        self.dataset = dataset
        logging.info('creating dataset finishes')

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

