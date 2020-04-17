"""
convert the raw structure data into RETAIN format: list of list indexes from 1
"""
import sys
sys.path.append('./')
import os
import pickle as pkl
from utils.vocab import load_vocab


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


def convert_data(train_PATH, valid_PATH, test_PATH,
                 word_vocab_PATH, position_vocab_PATH, tag_vocab_PATH, out_DIR):
    train_ys, train_codes, train_times = pkl.load(open(train_PATH, 'rb'))
    valid_ys, valid_codes, valid_times = pkl.load(open(valid_PATH, 'rb'))
    test_ys, test_codes, test_times = pkl.load(open(test_PATH, 'rb'))

    word_vocab = load_vocab(word_vocab_PATH)
    position_vocab_PATH = load_vocab(position_vocab_PATH)
    tag_vocab_PATH = load_vocab(tag_vocab_PATH, mode="tag")

    print("Vocabuary size for word, position and tags are {}, {}, {}".format(len(word_vocab), len(position_vocab_PATH),
                                                                             len(tag_vocab_PATH)))

    train_visits, train_labels, train_days = [],[],[]
    valid_visits, valid_labels, valid_days = [],[],[]
    test_visits, test_labels, test_days = [],[],[]

    train_counter = {"case":0, "control":0}
    valid_counter = {"case": 0, "control": 0}
    test_counter = {"case": 0, "control": 0}

    # traverse each train patient
    for index, (y, patient_codes, patient_times) in enumerate(zip(train_ys, train_codes, train_times)):
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

    pkl_train_visit_file = open(out_DIR + 'visits.train', 'wb')
    pkl_valid_visit_file = open(out_DIR + 'visits.valid', 'wb')
    pkl_test_visit_file = open(out_DIR + 'visits.test', 'wb')
    
    pkl_train_label_file = open(out_DIR + 'labels.train', 'wb')
    pkl_valid_label_file = open(out_DIR + 'labels.valid', 'wb')
    pkl_test_label_file = open(out_DIR + 'labels.test', 'wb')

    pkl_train_day_file = open(out_DIR + 'days.train', 'wb')
    pkl_valid_day_file = open(out_DIR + 'days.valid', 'wb')
    pkl_test_day_file = open(out_DIR + 'days.test', 'wb')


    print('In total converted {}, {}, {} train, valid, and test samples'.
          format(len(train_labels), len(valid_labels), len(test_labels)))
    print('In which the distributions of cases and controls for train, valid and test are\n\t{}\t{}\t{}\t{}\t{}\t{}'.
          format(train_counter['case'], train_counter['control'],
                 valid_counter['case'], valid_counter['control'],
                 test_counter['case'], test_counter['control']))

    pkl.dump(train_visits, pkl_train_visit_file, protocol=2)
    pkl.dump(valid_visits, pkl_valid_visit_file, protocol=2)
    pkl.dump(test_visits, pkl_test_visit_file, protocol=2)

    pkl.dump(train_labels, pkl_train_label_file, protocol=2)
    pkl.dump(valid_labels, pkl_valid_label_file, protocol=2)
    pkl.dump(test_labels, pkl_test_label_file, protocol=2)

    pkl.dump(train_days, pkl_train_day_file, protocol=2)
    pkl.dump(valid_days, pkl_valid_day_file, protocol=2)
    pkl.dump(test_days, pkl_test_day_file, protocol=2)


if __name__ == '__main__':
    data_dir = './data_samples/predictive_model/asthma/corpus/event_train_event_test/train_valid_test/'
    train_PATH = data_dir + 'adults/patient_dataset_train.fold1'
    valid_PATH = data_dir + 'adults/patient_dataset_valid.fold1'
    train_valid_PATH = data_dir + 'adults/patient_dataset_train'
    test_PATH = data_dir + 'adults/patient_dataset_test'
    word_vocab_PATH = data_dir + 'code_vocab.txt'
    position_vocab_PATH = data_dir + 'pos_vocab.txt'
    tag_vocab_PATH = data_dir + 'tag_vocab.txt'
    out_DIR = data_dir + 'adults/retain/fold1/'
    out_DIR = data_dir + 'adults/retain/all/'
    #convert_data(train_PATH, valid_PATH, test_PATH, word_vocab_PATH, position_vocab_PATH, tag_vocab_PATH, out_DIR)
    convert_data(train_valid_PATH, test_PATH, test_PATH, word_vocab_PATH, position_vocab_PATH, tag_vocab_PATH, out_DIR)


