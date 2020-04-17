"""

"""
import math
from typing import List

import numpy as np
# from DocumentFeatureSelection import interface
import pprint
import json
import sys
import pickle as pkl
from utils.visualization import draw_heatmap
from utils.cerner_utils import sort_dict
from utils.visualization import draw_code_time_distribution, draw_code_patientno_distribution


def json_load_patients(path):
    patients_info = []
    patient = []
    with open(path) as f:
        for line in f:
            if len(line.strip()) == 0:
                if len(patient) != 0:
                    patients_info.append(patient)
                    patient = []
            else:
                patient.append(json.loads(line))
    return patients_info


def get_visit_keywords_summarizer(visit_info, visit_weight, code_weights):
    """
    get the keywords for each visit based on codes and weights for each code
    :param visit_info:
    :param prediction_day: the day info for the prediction visit
    :param code_weights: weights for each code, a dict
    :param visit_weight: weight for the visit
    :return: a list of <keyword, time, weight> tuples for each visit
    """
    codes = visit_info['codes']
    len_words = len(codes)
    # happened within K months before
    day = visit_info['times'][2]
    month = int(day / 30)

    weight_tuples = []
    for i in range(len(codes)):
        code = codes[i]
        # only get weights for the first l codes, get rid of the paddings
        weight = code_weights[i] * visit_weight
        weight_tuples.append({'code': code, 'day': day, 'month': month, 'weight': weight})
    return weight_tuples


def get_patient_codes_summarizer(visits_keyword_tuples):
    """
    get the patient level keywords from visit-level and summarize
    :param visits_keyword_tuples: (code, day, month, weight)
    :return:
    """
    max_visit_len = np.max([len(tuples) for tuples in visits_keyword_tuples])
    weighted_visits = []
    patient_code_summarizer = {}

    # pad into max len for each visit by the day of visit[0] and weight 0.0
    def padded_visit(visit_tuples, max_len):
        l = max_len - len(visit_tuples)
        for _ in range(l):
            visit_tuples.append({'code': '', 'day': visit_tuples[0]['day'],
                                 'month': visit_tuples[0]['month'], 'weight': 0.0})
        return visit_tuples

    # sum up the weights for each timestamped code for each patient
    # print("visits keyword tuples len {}".format(len(visits_keyword_tuples)))
    for visit_keyword_tuples in visits_keyword_tuples:
        # print(visit_keyword_tuples)
        weighted_visits.append(padded_visit(visit_keyword_tuples, max_visit_len))
        for datum in visit_keyword_tuples:
            code, day, month, weight = datum['code'], datum['day'], datum['month'], datum['weight']
            if len(code) > 0:
                timed_code = code + '_' + str(month)
                if timed_code in patient_code_summarizer:
                    patient_code_summarizer[timed_code] += weight
                else:
                    patient_code_summarizer[timed_code] = weight
    # the first return param is for drawing heatmap after padding to consistent lengths
    # print("weighted visits length {}".format(len(weighted_visits)))
    return weighted_visits, patient_code_summarizer


def get_cohort_keywords_summarizer(patients_tags, patients_code_summarizer):
    """
    if a timestamped code occurs in more case patients but less control patients,
    the code is more important for case
    use a set of text feature selection methods to compute features
    :param patients_tags: case or control, the ground truth
    :param patients_code_summarizer:
    :return:
    """
    case_list, control_list = [], []
    for i in range(len(patients_tags)):
        type = patients_tags[i]
        # actually the code in summarizer is weighted, how to integrate weight further?
        patient_code_summarizer = patients_code_summarizer[i]
        patient = []
        for code, weight in patient_code_summarizer.items():
            patient.append({'code': code, 'weight': weight})
        if type == 1:
            case_list.append(patient)
        else:
            control_list.append(patient)

    print("size of case and control {}, {}".format(len(case_list), len(control_list)))
    cohort_keywords_summarizer = {
        "case": case_list,
        "control": control_list
    }
    return cohort_keywords_summarizer


# def compute_pmi(cohort_keywords_summarizer):
#     # print('cohort keywords', cohort_keywords_summarizer)
#     # input()
#     pmi_scored_object = interface.run_feature_selection(
#         input_dict=cohort_keywords_summarizer,
#         method='pmi',
#         n_jobs=1,
#         use_cython=False
#     )
#     pprint.pprint(pmi_scored_object.ScoreMatrix2ScoreDictionary())


def visits_keywords_screener(visits_keyword_tuples, threshold):
    """
    screen the visits keywords using certain criteria, e.g. weight threshold or
    :param visits_keyword_tuples:
    :param threshold:
    :return:
    """
    for i in range(len(visits_keyword_tuples)):
        removed_indices: List[int] = []
        for j in range(len(visits_keyword_tuples[i])):
            if visits_keyword_tuples[i][j]['weight'] < threshold:
                removed_indices.append(j)
        for j in removed_indices:
            del visits_keyword_tuples[i][j]
    return visits_keyword_tuples


""""""


def get_weighted_keywords(patients_tags, patients_preds, patients_probs, patients_codes_summarizer):
    """
    Wc = sum_d(p_d*w_c), p_d<[0,1], w_c<[0,1], for the case and control respectively
    :param patients_preds:
    :param patients_probs:
    :param patients_codes_summarizer: a dictionary contained timestamped code and weight
    :return:
    """
    Wc_case, Wc_control = {}, {}
    case_count, control_count = 0, 0
    # summarize weights for each word
    for y, pred, prob, patient_codes_summarizer in zip(patients_tags, patients_preds, patients_probs,
                                                       patients_codes_summarizer):
        for code in patient_codes_summarizer:
            w = prob[pred] * patient_codes_summarizer[code]
            if pred == 1:
                Wc_case[code] = Wc_case[code] + w if code in Wc_case else w
                case_count += 1
            else:
                Wc_control[code] = Wc_control[code] + w if code in Wc_control else w
                control_count += 1

    # output each word
    for code in Wc_case:
        Wc_case[code] /= case_count
    for code in Wc_control:
        Wc_control[code] /= control_count

    # for case and control, rank their keywords and list the weights

    Wc_case_sorted = sort_dict(Wc_case, 1)
    Wc_control_sorted = sort_dict(Wc_control, 1)
    for i in range(0, 200):
        print("Case keyword code {} and weight {}".format(Wc_case_sorted[i][0], Wc_case_sorted[i][1]))
        # input()


def get_features_with_high_df(patients_preds, patients_codes_summarizer, with_time=False):
    """
    get features that occur in multiple patients in a frequency decreasing order
    x-axis: #patients for features, y-axis: 10 high df features
    :param patients_preds:
    :param patients_codes_summarizer:
    :param with_time:
    :return:
    """
    # 1. get the count of patients for each code
    case_code_patient_counter, control_code_patient_counter = {}, {}
    for pred, patient_codes_summarizer in zip(patients_preds, patients_codes_summarizer):
        patient_codes_list = [code if with_time else code[0:code.rfind('_')]
                              for code in patient_codes_summarizer]
        patient_codes_set = set(patient_codes_list)
        for code in patient_codes_set:
            if pred == 0:
                case_code_patient_counter[code] = case_code_patient_counter[code] + 1 \
                    if code in case_code_patient_counter else 1
            else:
                control_code_patient_counter[code] = control_code_patient_counter[code] + 1 \
                    if code in control_code_patient_counter else 1

    # output the counter
    sorted_case_code_patient_counter = sort_dict(case_code_patient_counter, 1)
    sorted_control_code_patient_counter = sort_dict(control_code_patient_counter, 1)

    # print(len(sorted_case_code_patient_counter))

    # the most frequent occurred (multiple patients) timestamped code
    print("## Sorted code based on patient df with time stamp {}".format(with_time))
    for i in range(0, 100):
        code = sorted_case_code_patient_counter[i]
        print("Case keyword code {} and count of patients {}".format(code[0], code[1]))
    # for i in range(0,50):
    #     code = sorted_control_code_patient_counter[i]
    #     print("Control keyword code {} and count of patients {}".format(code[0], code[1]))

    return case_code_patient_counter, sorted_case_code_patient_counter


def get_code_feature_time_distribution(code, timed_case_code_patient_counter):
    """
    for a certain feature, see how different timestamped feature occur as keywords in patients
    x-axis: 0-12month, y-axis: #patients have w as a key feature
    :param patients_preds:
    :param patients_codes_summarizer:
    :return:
    """
    # only medication and diagnosis are considered
    print('code for func', code)
    if code.find('M_') < 0 and code.find('D_') < 0:
        print("{} not in code".format(code))
        return
    time_counters = []
    for i in range(0, 12):
        timestamped_code = code + '_' + str(i)
        # print(timestamped_code)
        time_counters.append(timed_case_code_patient_counter[timestamped_code]
                             if timestamped_code in timed_case_code_patient_counter else 0)
    return time_counters


def onelayer_statisticer(info_path, patients_preds, patients_probs, patients_weights):
    """
    process pmi analysis and find out the representative keywords for each patient and the cohort
    :param patients_infos: dict info for each patient
    :param patients_preds: predicted label for each patient
    :param patients_probs: two class probs for each patient
    :param patients_weights: weight for each timestamped code for each patient
    :return:
    """
    patients_infos = json_load_patients(info_path)  # pinfo and visits
    print("Loaded patients infos")
    patients_tags, patients_codes_summarizer = [], []
    patients_weighted_visits = []
    idx = 0
    for patient, pred, probs, weights in zip(patients_infos, patients_preds, patients_probs, patients_weights):
        # each patient
        if len(weights) < 1911:
            print("The {}th patient with max weights {}".format(idx, len(weights)))
        idx += 1
        sys.stdout.write((str(idx)) + '\r')
        sys.stdout.flush()
        pinfo, pvisits = patient[0], patient[1:]
        patients_tags.append(pinfo['case'])

        visits_keyword_tuples = []  # a list of list of code_tuples with variant visit length
        # get all codes and corresponding positions into a list of list
        for visit in pvisits:
            visits_keyword_tuples.append([{'code': code, 'day': visit['times'][2],
                                           'month': int(visit['times'][2] / 30), 'weight': 0.0}
                                          for code in visit['codes']])
            # print(visit['times'][2], int(visit['times'][2]/30))
            # input()
        counter = 0
        # print('len weights', len(weights))
        weight_sum = 0.0

        # input()
        # traverse each visit and assign weights to each code
        for i in range(len(visits_keyword_tuples)):
            # get weight from weights according to indices, the weights vector is padded at last
            for j in range(len(visits_keyword_tuples[i])):
                if counter >= len(weights):
                    print(pinfo)
                    break
                visits_keyword_tuples[i][j]['weight'] = weights[counter]
                weight_sum += weights[counter]
                counter += 1

        # screen the visit keywords using weights

        # the input parameter is padded in the func
        patient_weighted_visits, patient_codes_summarizer = get_patient_codes_summarizer(visits_keyword_tuples)

        # print(visits_keyword_tuples)
        patients_codes_summarizer.append(patient_codes_summarizer)
        patients_weighted_visits.append(patient_weighted_visits)
        print('Begin drawing')
        if pred == 1:
            draw_heatmap(patient_weighted_visits, probs[1], pred, pinfo['psk'])
        #input()

    #

    cohort_keywords_summarizer = get_cohort_keywords_summarizer(patients_tags, patients_codes_summarizer)
    # compute_pmi(cohort_keywords_summarizer)
    # weighting each code for different categories based on probabilities
    get_weighted_keywords(patients_preds, patients_probs, patients_codes_summarizer)
    # count the document frequencies for each ocde
    case_code_patient_counter, sorted_case_code_patient_counter = \
        get_features_with_high_df(patients_preds, patients_codes_summarizer, with_time=False)
    timed_case_code_patient_counter, timed_sorted_case_code_patient_counter = \
        get_features_with_high_df(patients_preds, patients_codes_summarizer, with_time=True)
    # get the time distribution for some top words
    get_time_distribution_for_top_codes(timed_case_code_patient_counter, sorted_case_code_patient_counter)
    return patients_weighted_visits, cohort_keywords_summarizer


def get_time_distribution_for_top_codes(timed_case_code_patient_counter, sorted_case_code_patient_counter):
    codes = []
    for i in range(0, 50):
        code_and_no = sorted_case_code_patient_counter[i]
        print("time distributed code {} happened on {} patients".format(code_and_no[0], code_and_no[1]))
        time_counters = get_code_feature_time_distribution(code_and_no[0], timed_case_code_patient_counter)
        print("distribution:\t", time_counters)
        codes.append(code_and_no[0])
    return codes


def twolayer_statisticer(train_infopath, test_infopath, patients_preds, patients_probs, patients_visits_weights, patients_codes_weights, outdir):
    """

    :param info_path:
    :param patients_preds:
    :param patients_probs:
    :param patients_visits_weights: [list of patients, list of visits]
    :param patients_codes_weights: [list of patients, list of codes]
    :return:
    """
    train_patients_infos = json_load_patients(train_infopath)  # pinfo and visits
    test_patients_infos = json_load_patients(test_infopath)  # pinfo and visits
    patients_infos = train_patients_infos + test_patients_infos
    print("Loaded patients infos")
    patients_tags, patients_codes_summarizer = [], []
    patients_weighted_visits = []
    patients_visits_keywords_tuples = []
    # traverse each patient
    for patient, pred, probs, patient_visits_weights, patient_codes_weights in \
            zip(patients_infos, patients_preds, patients_probs, patients_visits_weights, patients_codes_weights):
        # each patient
        pinfo, pvisits = patient[0], patient[1:]
        # print("pvisits len {}".format(len(pvisits)))
        patients_tags.append(pinfo['case'])
        visits_keyword_tuples = []  # a list of list of code_tuples with variant visit length
        # get all codes and corresponding positions into a list of list
        # for visit in pvisits:
        #     visits_keyword_tuples.append([{'code':code, 'day': visit['times'][2],
        #                                    'month':int(visit['times'][2]/30), 'weight':0.0}
        #                                     for code in visit['codes']])

        # print("visits keyword tuples init len {}".format(len(visits_keyword_tuples)))
        counter = 0
        weight_sum = 0.0
        for i in range(len(pvisits)):
            visit_weight = patient_visits_weights[i]  # 1d: list of visits
            codes_weights = patient_codes_weights[i]  # 2d: list of visits, list of codes
            visit_info = pvisits[i]
            visit_keyword_tuples = get_visit_keywords_summarizer(visit_info, visit_weight, codes_weights)
            # print(visit_keyword_tuples)
            # input()
            visits_keyword_tuples.append(visit_keyword_tuples)
            counter += len(codes_weights)
            weight_sum += np.sum(codes_weights)

        patients_visits_keywords_tuples.append(visits_keyword_tuples)
        # print("input visits keyword tuples len {}".format(len(visits_keyword_tuples)))
        # print(visits_keyword_tuples)
        # input()
        patient_weighted_visits, patient_codes_summarizer = get_patient_codes_summarizer(visits_keyword_tuples)
        # print("output patient weighted visits len {}".format(len(patient_weighted_visits)))
        # print(patient_weighted_visits)
        # input()

        ### draw heatmap for each patient
        #print('draw_heatmap')
        #if pred == 1 and probs[1] > 0.5:
        #    draw_heatmap(patient_weighted_visits, probs[1], pred, pinfo['psk'], outdir + 'heatmap/')
            # input()
        patients_weighted_visits.append(patient_weighted_visits)
        # print(visits_keyword_tuples)
        patients_codes_summarizer.append(patient_codes_summarizer)
        # input()

    # based on the ground truth
    # cohort_keywords_summarizer = get_cohort_keywords_summarizer(patients_tags, patients_codes_summarizer)
    # compute_pmi(cohort_keywords_summarizer)

    # weighting each code for different categories based on probabilities
    get_weighted_keywords(patients_tags, patients_preds, patients_probs, patients_codes_summarizer)
    
    print('pause')
    #input()  
    # count the document frequencies for each code
    case_code_patient_counter, sorted_case_code_patient_counter = \
        get_features_with_high_df(patients_preds, patients_codes_summarizer, with_time=False)
    # the timestamped code
    timed_case_code_patient_counter, timed_sorted_case_code_patient_counter = \
        get_features_with_high_df(patients_preds, patients_codes_summarizer, with_time=True)
    # get the time distribution for some top words
    codes = get_time_distribution_for_top_codes(timed_case_code_patient_counter, sorted_case_code_patient_counter)
    for i,code in enumerate(codes):
        if i > 50:
            break
        if code[0:2] == 'D_' or code[0:2] == 'M_':
            if code.find('D_493') >= 0:
                continue
            # pure_code = code[0:code.rfind('_')]
            # print("code {} and pure code {}".format(code, pure_code))
            print("code for drawing time distribution is {}".format(code))
            #draw = input('1 or 0?')
            #if draw == '1':
            draw_code_time_distribution(code, patients_visits_keywords_tuples, outdir + 'codetime/')
    
    return patients_weighted_visits  # cohort_keywords_summarizer


def data_statistics(info_path):
    patients_infos = json_load_patients(info_path)  # pinfo and visits
    print("Loaded patients infos")

    max_visit_len, max_code_len, min_visit_len, min_code_len = 0, 0, 10000, 10000
    case_gender_counter, case_race_counter, control_gender_counter, control_race_counter = {}, {}, {}, {}
    age_index_counters, age_exacer_counters = [], []
    visit_lens, visit_code_lens = [], []
    for patient in patients_infos:
        pinfo, pvisits = patient[0], patient[1:]
        gender, race = pinfo['gender'], pinfo['race']
        race = 'Other' if race in {"NULL", "Null", "Unknown", "Not Mapped"} else race

        age_index, age_exacer = pvisits[0]['age'], pvisits[-1]['age']
        age_index_counters.append(age_index)
        age_exacer_counters.append(age_exacer)
        if pinfo['case'] == 1:
            case_gender_counter[gender] = case_gender_counter[gender] + 1 if gender in case_gender_counter else 1
            case_race_counter[race] = case_race_counter[race] + 1 if race in case_race_counter else 1
        else:
            control_gender_counter[gender] = control_gender_counter[
                                                 gender] + 1 if gender in control_gender_counter else 1
            control_race_counter[race] = control_race_counter[race] + 1 if race in control_race_counter else 1
        visit_len = len(pvisits)
        if visit_len > max_visit_len:
            max_visit_len = visit_len
        if visit_len < min_visit_len:
            min_visit_len = visit_len
        visit_lens.append(visit_len)
        for visit in pvisits:
            code_len = len(visit['codes'])
            if code_len < min_code_len:
                min_code_len = code_len
            if code_len > max_code_len:
                max_code_len = code_len
            visit_code_lens.append(code_len)

    ## print
    print("Case gender distributions:\n")
    for gender in case_gender_counter:
        print("{}:\t{}".format(gender, case_gender_counter[gender]))
    print("Case race distribution:\n")
    for race in case_race_counter:
        print("{}:\t{}".format(race, case_race_counter[race]))
    print("Control gender distributions:\n")
    for gender in control_gender_counter:
        print("{}:\t{}".format(gender, control_gender_counter[gender]))
    print("Control race distribution:\n")
    for race in control_race_counter:
        print("{}:\t{}".format(race, control_race_counter[race]))
    print("Max min and average number of visits:\t {} {} {}".format(max_visit_len, min_visit_len,
                                                                    np.average(np.array(visit_lens).astype(np.int))))
    print("The std of number of visits:\t {}".format(np.std(np.array(visit_lens).astype(np.int))))
    print("Max min and avarage number of code per visit:\t {} {} {}".format(max_code_len, min_code_len, np.average(
        np.array(visit_code_lens).astype(np.int))))
    print("The std of number of codes:\t {}".format(np.std(np.array(visit_code_lens).astype(np.int))))
    print("Avarage age of asthma index and exacerbation :\t{} {}".format(
        np.average(np.array(age_index_counters).astype(np.int)),
                   np.average(np.array(age_exacer_counters).astype(np.int))))
