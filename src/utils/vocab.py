"""
build_vocab
"""
from pathlib import Path
from collections import Counter
import string
import sys

PAD_WORD = '</s>'
PAD_TAG = 'O'

def build_vocab(X, MINCOUNT=1, encoding=sys.stdout.encoding):
    counter_words = Counter()
    for words in X:
        # print(words)
        new_words = []
        for word in words:
            # convert a bytes like object to string, and remove punctuations
            # t = str(word,sys.stdout.encoding).translate(str.maketrans("", "", string.punctuation))
            # t = str(word, encoding)
            # for a string not byte
            t = str(word)
            if t != '':
                new_words.append(t)
        counter_words.update(new_words)
    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}
    return vocab_words

def load_vocab(path, mode='word'):
    vocab = {}
    index = 0
    with Path(path).open('r') as f:
        for l in f:
            key = l.strip() if mode == 'word' else int(l.strip())
            vocab[key] = index
            index += 1
    return vocab



