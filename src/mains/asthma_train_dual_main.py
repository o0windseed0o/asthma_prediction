"""
Function: the main function for training TSANN model for asthma exacerbation prediction
Author: Xiang.Yang
Contact: xiangyang.hitsz@gmail.com
Date created: 04/16/2020
"""
import tensorflow as tf
import sys

sys.path.append('./')

from data_loader.EHRTwoLayerLoader import EHRTwoLayerLoader as DataLoader
from models.tsann_model import TSANN as Model
from trainers.pred_trainer import PredTrainer
from utils.config import get_config_from_json, update_config_by_summary
from utils.dirs import create_dirs, remove_dir
from utils.logger import Logger
from utils.utils import get_args
from utils.vocab import load_vocab

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # flexibly show the tf logs
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use GPU 0


def main():
    # capture the config path from the runtime arguments
    # then process the json configuration file
    args = get_args()
    print("Reading config from {}".format(args.config))
    config, _ = get_config_from_json(args.config)
    # add summary and model directory
    config = update_config_by_summary(config)          
    # if to remove the previous results, set -d 1 as a parameter
    print('Whether to del the previous saved model', args.delete)
    if args.delete == '1':
        # delete existing checkpoints and summaries
        print('Deleting existing models and logs from:')
        print(config.summary_dir, config.checkpoint_dir)
        remove_dir(config.summary_dir)
        remove_dir(config.checkpoint_dir)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    """Load data"""
    # load global word, position and tag vocabularies
    word_vocab = load_vocab(path=config.datadir + config.word_vocab_path, mode='word')
    position_vocab = load_vocab(path=config.datadir + config.pos_vocab_path, mode='pos')
    tag_vocab = load_vocab(path=config.datadir + config.tag_vocab_path, mode='tag')
    config.word_vocab_size = len(word_vocab)
    config.pos_vocab_size = len(position_vocab)
    config.tag_vocab_size = len(tag_vocab)

    print('word vocab size:', config.word_vocab_size)

    # create your data generator to load train data
    x_path = config.datadir + config.train_path
    train_loader = DataLoader(config, x_path, word_vocab, position_vocab, tag_vocab)
    train_loader.load_data()
    # update the max length for each patient and each visit to be used in lstm
    train_max_patient_len = train_loader.max_patient_len
    train_max_visit_len = train_loader.max_visit_len

    # create your data generator to load valid data
    x_path = config.datadir + config.valid_path
    valid_loader = DataLoader(config, x_path, word_vocab, position_vocab, tag_vocab)
    valid_loader.load_data()
    valid_max_patient_len = valid_loader.max_patient_len
    valid_max_visit_len = valid_loader.max_visit_len

    # create your data generator to load test data
    x_path = config.datadir + config.test_path
    test_loader = DataLoader(config, x_path, word_vocab, position_vocab, tag_vocab)
    test_loader.load_data()
    test_max_patient_len = test_loader.max_patient_len
    test_max_visit_len = test_loader.max_visit_len

    print("The max patient lengths for train, valid and test are {}, {}, {}"
          .format(train_max_patient_len, valid_max_patient_len, test_max_patient_len))
    print("The max visit lengths for train, valid and test are {}, {}, {}"
          .format(train_max_visit_len, valid_max_visit_len, test_max_visit_len))

    # select the maximum lengths of visits and codes as the size of lstm
    config.max_patient_len = max([train_max_patient_len, valid_max_patient_len, test_max_patient_len])
    config.max_visit_len = max([train_max_visit_len, valid_max_visit_len, test_max_visit_len])
    
    train_loader.pad_data(config.max_patient_len, config.max_visit_len)
    valid_loader.pad_data(config.max_patient_len, config.max_visit_len)
    test_loader.pad_data(config.max_patient_len, config.max_visit_len)

    # add num_iter_per_epoch to config for trainer
    config.train_size = train_loader.get_datasize()
    config.valid_size = valid_loader.get_datasize()
    config.test_size = test_loader.get_datasize()
    config.num_iter_per_epoch = int(config.train_size / config.batch_size)
    print("The sizes for train, valid and test are {}, {}, {}"
          .format(config.train_size, config.valid_size, config.test_size))

    """Run model"""
    # create tensorflow session
    # specify only using one GPU
    tfconfig = tf.ConfigProto(device_count={'GPU': 1})
    # allow the dynamic increase of GPU memory
    tfconfig.gpu_options.allow_growth = True
    # limit the maximum of GPU usage as 0.5
    #tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=tfconfig) as sess:
        # create an instance of the model you want
        model = Model(config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = PredTrainer(sess, model, train_loader, valid_loader, test_loader, config, logger)
        # load model if exists
        model.load(sess)
        # here you train your model
        trainer.train()

    # testers


if __name__ == '__main__':
    main()
