import sys
sys.path.append('./')
import os
import pickle as pkl
from utils.vocab import load_vocab
import numpy as np
import random

def make_visit(codes, word_vocab):
    l = len(word_vocab)
    vec = np.zeros((l,), dtype=float)
    indices = []
    # print(codes)
    for code in codes:
        # print(code)
        idx = word_vocab[code] if code in word_vocab else word_vocab['</s>']
        indices.append(idx)
    vec[indices] = 1.
    # print(word_vocab)
    # print(vec)
    # print(np.sum(vec))
    return vec

def make_visits(patient_codes, word_vocab):
    visits = []
    for codes in patient_codes:
        visits.append(make_visit(codes, word_vocab))
    return visits


def make_days(patient_times):
    days = []
    #print(len(patient_times))
    #print(len(patient_times))
    for time_vec in patient_times:
        days.append(float(time_vec[1]))
    #print(days)

    #input()
    return days


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

    train_buckets, valid_buckets, test_buckets = {}, {}, {}

    for index, (y, patient_codes, patient_times) in enumerate(zip(train_ys, train_codes, train_times)):
        # print(patient_codes)
        visits = make_visits(patient_codes, word_vocab)
        label = [0.0, 1.0] if y == 1 else [1.0, 0.0]
        elapse = make_days(patient_times)
        l = len(visits)
        #print("buck size:", l)
        #print(elapse)
        #input()
        if l in train_buckets:
            train_buckets[l].append([visits, elapse, label])
        else:
            train_buckets[l] = [[visits, elapse, label]]

    # test
    for index, (y, patient_codes, patient_times) in enumerate(zip(test_ys, test_codes, test_times)):
        visits = make_visits(patient_codes, word_vocab)
        label = [0.0, 1.0] if y == 1 else [1.0, 0.0]
        elapse = make_days(patient_times)
        l = len(visits)
        if l in test_buckets:
            test_buckets[l].append([visits, elapse, label])
        else:
            test_buckets[l] = [[visits, elapse, label]]

    # set a batch_size, if a bucket contains > batch_size, then split it into multiple batches, otherwise one batch
    batch_size = 32
    for i in train_buckets:
        bucket = train_buckets[i]
        size = len(bucket)
        if size > batch_size:
            base = int(size/batch_size)
            num_of_batch = base + 1 if size % batch_size != 0 else base
            for j in range(0, num_of_batch):
                batch = bucket[j*batch_size:np.min([size,(j+1)*batch_size])]
                train_visits.append([datum[0] for datum in batch])
                train_days.append([datum[1] for datum in batch])
                train_labels.append([datum[2] for datum in batch])
        else:
            batch = bucket
            train_visits.append([datum[0] for datum in batch])
            train_days.append([datum[1] for datum in batch])
            train_labels.append([datum[2] for datum in batch])

    for i in test_buckets:
        bucket = test_buckets[i]
        size = len(bucket)
        if size > batch_size:
            base = int(size/batch_size)
            num_of_batch = base + 1 if size % batch_size != 0 else base
            for j in range(0, num_of_batch):
                batch = bucket[j*batch_size:np.min([size,(j+1)*batch_size])]
                test_visits.append([datum[0] for datum in batch])
                test_days.append([datum[1] for datum in batch])
                test_labels.append([datum[2] for datum in batch])
        else:
            batch = bucket
            test_visits.append([datum[0] for datum in batch])
            test_days.append([datum[1] for datum in batch])
            test_labels.append([datum[2] for datum in batch])

    train_data_path = out_DIR + 'data_train.pkl'
    test_data_path = out_DIR + 'data_test.pkl'
    train_day_path = out_DIR + 'elapsed_train.pkl'
    test_day_path = out_DIR + "elapsed_test.pkl"
    train_label_path = out_DIR + 'label_train.pkl'
    test_label_path = out_DIR + 'label_test.pkl'

    train_data = list(zip(train_visits, train_days, train_labels))
    random.shuffle(train_data)
    train_visits, train_days, train_labels = zip(*train_data)

    print("train size and test size", len(train_labels), len(test_labels))

    pkl.dump(np.array(train_visits), open(train_data_path, 'wb'), protocol=2)
    pkl.dump(np.array(test_visits), open(test_data_path, 'wb'), protocol=2)
    pkl.dump(np.array(train_days), open(train_day_path, 'wb'), protocol=2)
    pkl.dump(np.array(test_days), open(test_day_path, 'wb'), protocol=2)
    pkl.dump(np.array(train_labels), open(train_label_path, 'wb'), protocol=2)
    pkl.dump(np.array(test_labels), open(test_label_path, 'wb'), protocol=2)


if __name__ == '__main__':
    train_PATH = './data_samples/predictive_model/asthma/adults/patient_dataset_train.fold5'
    valid_PATH = './data_samples/predictive_model/asthma/adults/patient_dataset_test.fold5'
    test_PATH = './data_samples/predictive_model/asthma/adults/patient_dataset_test.fold5'
    word_vocab_PATH = './data_samples/predictive_model/asthma/word_vocab.txt'
    position_vocab_PATH = './data_samples/predictive_model/asthma/pos_vocab.txt'
    tag_vocab_PATH = './data_samples/predictive_model/asthma/tag_vocab.txt'
    out_DIR = './data_samples/predictive_model/asthma/adults_tlstm/fold5/'
    convert_data(train_PATH, valid_PATH, test_PATH, word_vocab_PATH, position_vocab_PATH, tag_vocab_PATH, out_DIR)
