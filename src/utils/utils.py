import argparse
import json
from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-d', '--delete',
        metavar='D',
        default='0',
        help='To delete the previous checkpoints and summaries'
    )
    args = argparser.parse_args()
    return args


def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def f1(y_true, y_preds):
    return f1_score(y_true, y_preds)

def prf(y_true, y_preds):
    p,r,f,s = precision_recall_fscore_support(y_true, y_preds, average='macro')
    return p, r, f


def auc_value(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=1)
    return auc(fpr, tpr)


def cmatrix(y_true, y_preds):
    return confusion_matrix(y_true, y_preds)