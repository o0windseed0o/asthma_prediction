import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def update_config_by_summary(config):
    config.summary_dir = os.path.join(config.model_dir, "summary/")
    config.checkpoint_dir = os.path.join(config.model_dir, "checkpoint/")
    return config


def update_config_by_vocab(config, word_vocab_size, tag_vocab_size):
    config.word_vocab_size = word_vocab_size
    config.tag_vocab_size = tag_vocab_size
    return config


