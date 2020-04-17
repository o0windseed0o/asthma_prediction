"""
convert the raw structure data into the sklearn format for lr or random forest
    
"""
import pickle as pkl
import sys
sys.path.append('./')
import os
from utils.vocab import load_vocab
import numpy as np
from scipy.sparse import csr_matrix


def make_visits(patient_codes, vocab):
    patient_visits = []
    for visit_codes in patient_codes:
        visit = []
        for code in visit_codes:
            idx = vocab[code] if code in vocab else 0
            if code not in vocab:
                print(code)
            visit.append(idx)
        patient_visits.append(visit)
    return patient_visits


def make_days(patient_times):
    patient_days = []
    for time in patient_times:
        patient_days.append(time[2])
    return patient_days


def make_patient(patient_codes, patient_times, feature_template, flag='notime'):
    x = {}
    #print(feature_template)
    till_patient_times = [patient_time[2] for patient_time in patient_times]
    for visit_codes, patient_time in zip(patient_codes, till_patient_times):
        for code in visit_codes:
            #print(code)
            if flag == 'notime':
                index = feature_template[code]
            else:
                code = code + '-' + str(patient_time)
                if code not in feature_template:
                    print(code)
                    print(patient_times)
                    input()
                else:
                    index = feature_template[code] ############
            if code in x:
                #x[code] = 1
                x[code] += 1
            else:
                x[code] = 1
    #print(x)
    x_li = []
    for code in feature_template:
        if code in x:
            x_li.append(x[code])
        else:
            x_li.append(0)
    #print(x_li)
    #input()
    return x_li    



def make_vocab(train_codes, valid_codes, test_codes):
    index = 1
    vocab = {}
    for patient_codes in train_codes + valid_codes + test_codes:
        for visit_codes in patient_codes:
            for code in visit_codes:
                if code in vocab:
                    continue
                else:
                    vocab[code] = index
                    index += 1
    return vocab


def make_feature_template(word_vocab, position_vocab, flag='notime'):
    feature_template = {}
    index = 1
    for word in word_vocab:
        if flag == 'notime':
            feature_template[word] = index
            index += 1
        else:
            for day in position_vocab:
                feature_template[word + '-' + str(day)] = index ######## try stratify into month later
                index += 1
    return feature_template


def convert_data(train_PATH, test_PATH,
                 word_vocab_PATH, position_vocab_PATH, tag_vocab_PATH, out_DIR, flag='notime'):
    train_ys, train_codes, train_times = pkl.load(open(train_PATH, 'rb'))
    test_ys, test_codes, test_times = pkl.load(open(test_PATH, 'rb'))

    word_vocab = load_vocab(word_vocab_PATH)
    position_vocab = load_vocab(position_vocab_PATH)
    tag_vocab = load_vocab(tag_vocab_PATH, mode="tag")

    print("Vocabuary size for word, position and tags are {}, {}, {}".format(len(word_vocab), len(position_vocab),
                                                                             len(tag_vocab)))

    feature_template = make_feature_template(word_vocab, position_vocab, flag)
    print(f"the feature number for lr is {len(feature_template)}")

    train_visits, train_labels, train_days = [],[],[]
    test_visits, test_labels, test_days = [],[],[]
    
    train_X, valid_X, test_X = [], [], []
    train_Y, valid_Y, test_Y = [], [], []
   
    # traverse each train patient
    for index, (y, patient_codes, patient_times) in enumerate(zip(train_ys, train_codes, train_times)):
        x = make_patient(patient_codes, patient_times, feature_template, flag)
        train_X.append(x)
        train_Y.append(y)
    for index, (y, patient_codes, patient_times) in enumerate(zip(test_ys, test_codes, test_times)):
        x = make_patient(patient_codes, patient_times, feature_template, flag)
        test_X.append(x)
        test_Y.append(y)

    train_X = csr_matrix(np.asarray(train_X))
    test_X = csr_matrix(np.asarray(test_X))

    '''
        train_visits.append(make_visits(patient_codes, word_vocab))
        train_labels.append(y)
        key = "case" if y == 1 else "control"
        train_counter[key] += 1
        train_days.append(make_days(patient_times))

    # valid    
    for index, (y, patient_codes, patient_times) in enumerate(zip(valid_ys, valid_codes, valid_times)):
        valid_visits.append(make_visits(patient_codes, word_vocab))
        valid_labels.append(y)
        key = "case" if y == 1 else "control"
        valid_counter[key] += 1
        valid_days.append(make_days(patient_times))

    # test
    for index, (y, patient_codes, patient_times) in enumerate(zip(test_ys, test_codes, test_times)):
        test_visits.append(make_visits(patient_codes, word_vocab))
        test_labels.append(y)
        key = "case" if y == 1 else "control"
        test_counter[key] += 1
        test_days.append(make_days(patient_times))
    '''
    pkl.dump(train_X, open(out_DIR + 'lr_time_day.trainX', 'wb'))
    pkl.dump(test_X, open(out_DIR + 'lr_time_day.testX', 'wb'))
    #pkl.dump(train_X, open(out_DIR + 'lr_notime.trainX', 'wb'))
    #pkl.dump(test_X, open(out_DIR + 'lr_notime.testX', 'wb'))
    pkl.dump(np.asarray(train_Y), open(out_DIR + 'lr.trainY', 'wb'))
    pkl.dump(np.asarray(test_Y), open(out_DIR + 'lr.testY', 'wb'))
    print("output data finished")    

if __name__ == '__main__':
    data_dir = './data_samples/predictive_model/asthma/corpus/event_previous_25_train_index_test_4/train_valid_test/'
    train_PATH = data_dir + 'adults/patient_dataset_train'
    test_PATH = data_dir + 'adults/patient_dataset_test'
    word_vocab_PATH = data_dir + 'code_vocab.txt'
    position_vocab_PATH = data_dir + 'pos_vocab.txt'
    tag_vocab_PATH = data_dir + 'tag_vocab.txt'
    out_DIR = data_dir + 'adults/lr/all/'
    convert_data(train_PATH, test_PATH, word_vocab_PATH, position_vocab_PATH, tag_vocab_PATH, out_DIR, flag='time')

