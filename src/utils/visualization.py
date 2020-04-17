from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import random

def draw_auc_curve(testY, probs):
    """
    plot auroc
    :param testY:
    :param probs:
    :return:
    """
    # print (probs)
    fpr, tpr, thresholds = roc_curve(testY, [prob[1] for prob in probs], pos_label=1)
    auc_value = auc(fpr, tpr)
    # plot the roc curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def draw_pr_curve(testY, probs):
    """
    plot the pr curve
    :param testY:
    :param probs:
    :return:
    """
    ps, rs = [], []
    for threshold in np.arange(0.0, 1.0, 0.01):
        pred = []
        for prob0, prob1 in probs:
            if prob1 > threshold:
                pred.append(1)
            else:
                pred.append(0)
        ps.append(precision_score(testY, pred))
        rs.append(recall_score(testY, pred))
    plt.figure()
    lw = 2
    plt.plot(ps, rs, color='red',
             lw=lw, label='PR curve')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR curve')
    plt.legend(loc="lower right")
    plt.show()


def draw_heatmap(weighted_visits, prob, pred, psk, outdir):
    weights = [[datum['weight'] for datum in visit] for visit in weighted_visits]
    codes = [[datum['code'] for datum in visit] for visit in weighted_visits]
    days = [visit[0]['day'] for visit in weighted_visits]
    annotations = np.asarray(codes)
    max_len = np.max(np.asarray([len(visits) for visits in codes]))
    xticklabels = ['code-' + str(i) for i in range(max_len)]
    yticklabels = ['v' + str(i) + '(-' + str(int(days[i])) + 'd)' for i in range(len(days))]
    ax = plt.axes()

    heatmap = sns.heatmap(weights, fmt='', cmap="YlGnBu", annot=annotations,
                          annot_kws={"size": 5}, xticklabels=xticklabels,
                          yticklabels=yticklabels, ax=ax, cbar=True)
    ax.set_title('heatmap for predicting {0} with risk proba {1:.4f}'.format(pred, prob))
    #plt.figure(dpi=500)
    #plt.show()
    figpath = outdir + psk + '.jpg'
    plt.savefig(figpath, format='jpg', dpi=600)

# each top code and its corresponding weight and time
def draw_code_time_distribution(code_, patients_visits_keywords_tuples, outdir):
    """

    :param patients_visits_keywords_tuples: [code, day, month, weight]
    :return:
    """
    # weighted_keywords = [] # x:day, y:random height between [0,1], value: code weights in each patient
    day_l, month_l, height_l, weight_l = [], [], [], []
    # traverse each patient
    for patient_visits_keyword_tuples in patients_visits_keywords_tuples:
        for visit_keyword_tuples in patient_visits_keyword_tuples:
            for datum in visit_keyword_tuples:
                code, day, month, weight = datum['code'], datum['day'], datum['month'], datum['weight']
                print(code, day, month, weight)
                #input()
                weight = round(weight + 0.001, 2)*1000
                # weight = '{:.2f}'.format(weight)
                if code == code_ and float(weight) >= 0.001:
                    day_l.append((day + np.random.uniform(-0.5,0.5))*-1)
                    month_l.append(month)
                    weight_l.append(weight)
                    height_l.append(np.random.uniform(0.0,1.0))
    weighted_keywords = pd.DataFrame({'day': day_l, 'month':month_l, 'weight':weight_l, 'height': height_l})
    print("size of population:", weighted_keywords.size)
    if weighted_keywords.size > 10000:
        weighted_keywords = weighted_keywords[0:2000]

    # print(df[0:10])
    # input()
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    ax = sns.scatterplot(x="day", y="height", size="weight", hue="weight", alpha=.8, palette=cmap,
            legend=False, data = weighted_keywords)
    ax.set_ylabel('patient node')
    ax.set_title('Time distribution among patients for factor {}'.format(code_))
    ax.set(yticklabels=[])
    #plt.show()
    figpath = outdir + code_ + '.jpg'
    plt.savefig(figpath, format='jpg', dpi=600)

# bar chart, each top code correlates to how many people
def draw_code_patientno_distribution(timed_case_code_patient_counter):
    """

    :param timed_case_code_patient_counter:
    :return:
    """
    pass

if __name__ == '__main__':
    # visualization test
    tips = sns.load_dataset("tips")
    print(tips)
    input()
    ax = sns.scatterplot(x="total_bill", y="tip", size="size", data = tips)
    plt.show()
