import numpy as np
import numpy.random as random
import pickle as pkl

code_vocab = ['</s>', 'A', 'B', 'C', 'D', 'E']
pos_vocab = [str(i) for i in range(10)]
tag_vocab = ['0', '1']


def save_vocab(code_vocab_path, pos_vocab_path, tag_vocab_path):
    with open(code_vocab_path, 'w') as fout:
        for code in code_vocab:
            fout.write(code + '\n')
    with open(pos_vocab_path, 'w') as fout:
        for pos in pos_vocab:
            fout.write(pos + '\n')
    with open(tag_vocab_path, 'w') as fout:
        for tag in tag_vocab:
            fout.write(tag + '\n')


def generate_samples(path, size):
    tags, patients, positions = [], [], []

    for i in range(size):
        # sampel the tag
        tag_index = random.randint(low=0, high=len(tag_vocab), size=1)[0]
        tag = tag_vocab[tag_index]
        # sample the number of visits 
        num_visits = random.randint(low=1, high=5, size=1)[0]
        # sample the time of visits
        times_visits = random.randint(low=1, high=len(pos_vocab), size=num_visits-1)
        times_visits = np.append(times_visits, [0])
        times_visits.sort()
        times_visits = times_visits
        time_start, time_end = 0, 0
        codes, times = [], []
        for j in range(num_visits):
            # sample the number of codes
            num_codes = random.randint(low=1, high=len(code_vocab)-1, size=1)[0]
            code_positions = random.randint(len(code_vocab), size=num_codes)
            visit = []
            for p in code_positions:
                visit.append(code_vocab[p])
            codes.append(visit)
            # calculate time
            time_gap = times_visits[j] - times_visits[j-1] if j > 0 else 0
            times.append([times_visits[j], time_gap, times_visits[-1]-times_visits[j]])
        tags.append(tag)
        patients.append(codes)
        positions.append(times)
    pkl.dump([tags, patients, positions], open(path, 'wb'))
   
def generate_samples_for_all(train_path, valid_path, test_path):
    # 5 patients for training
    # 2 patients for validating
    # 2 patients for testing
    generate_samples(train_path, 15)
    generate_samples(valid_path, 5)
    generate_samples(test_path, 5)


def main():
    # declare the path for data generation
    data_dir = '../data/sample/'
    train_path = data_dir + 'sample.train'
    valid_path = data_dir + 'sample.valid'
    test_path = data_dir + 'sample.test'
    code_vocab_path = data_dir + 'code_vocab.txt'
    pos_vocab_path = data_dir + 'pos_vocab.txt'
    tag_vocab_path = data_dir + 'tag_vocab.txt'
    
    # save vocab
    save_vocab(code_vocab_path, pos_vocab_path, tag_vocab_path)
    # generate samples
    generate_samples_for_all(train_path, valid_path, test_path)

if __name__ == '__main__':
    main()
