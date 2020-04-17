__author__ = 'xiangyang'
import os
from datetime import datetime
import pickle as pkl


# traverse a directory
def walk_dir(path):
    path_list = []
    for root, dirs, files in os.walk(path):
        for file_ in files:
            path_list.append(os.path.join(root, file_))
    return path_list

# if is a case patient, return the final diagnosis code
# else, return ''
def is_case_patient(patient, codelist):
    for visit in patient:
        code = contain_dignosis(visit, codelist)
        if code != '':
            return code
    return ''

# if a patient contains a diagnosis for asthma
def contain_dignosis(visit, code_list):
    for encounter_datum in visit:
        for code in code_list:
            if code in encounter_datum['diagnosis_code']:
                return code
    else:
        return ''

# convert a string type to a datetime type
def convert_to_datetime(string):
    date = string[0:string.find(' ')]
    dt = datetime.strptime(date, '%Y-%m-%d')
    return dt

# the last time subs the first time visiting the hospital
def patient_duration(visit_list):
    date_b = convert_to_datetime(visit_list[0][0]['admitted_dt_tm'])
    date_e = convert_to_datetime(visit_list[-1][0]['admitted_dt_tm'])
    days = str(date_e - date_b)
    if days.find('day') >= 0:
        days = days[0:days.find(' day')]
    else:
        days = 0
    return int(days)

# minus between two time exp
def time_minus(dt_tm_e, dt_tm_b):
    date_e = convert_to_datetime(dt_tm_e)
    date_b = convert_to_datetime(dt_tm_b)
    days = str(date_e - date_b)
    if days.find('day') >= 0:
        days = days[0:days.find(' day')]
    else:
        days = 0
    return int(days)

# sort a dictionary
def sort_dict(x, dim=1):
    sorted_dict = sorted(x.items(), key=lambda x: x[dim], reverse=True)
    new_list = []
    for key, value in sorted_dict:
        new_list.append([key, value])
    return new_list

# return an intersection set of two dictionaries
def intersection(dict_a, dict_b):
    intersection = {x: dict_a[x] for x in dict_a if x in dict_b}
    return intersection


"""
For code mapping
"""
# icd code mapping
def read_icd10icd9_mapping(PATH):
    icd102icd9 = {}
    with open(PATH, 'r') as f:
        for l in f:
            if len(l.strip()) > 0:
                vec = l.strip().split()
                icd10, icd9 = normalize_icd10(vec[0]), normalize_icd9(vec[1])
                # print (icd10, icd9)
                # input()
                if icd10 not in icd102icd9:
                    icd102icd9[icd10] = icd9
    print ('Loaded %d mappings from icd10 to icd9' % len(icd102icd9))
    return icd102icd9


# add . to build icd9 code
def normalize_icd9(code):
    dot_position = 4 if code[0] == 'E' else 3
    if len(code) == dot_position:
        return code
    else:
        return code[0:dot_position] + '.' + code[dot_position:]


# add . to build icd10 code
def normalize_icd10(code):
    if len(code) == 3:
        return code
    else:
        return code[0:3] + '.' + code[3:]

"""
For code grouping
"""
def read_ccs_icd9_multi_mapping(PATH):
    code2id = {}
    codes = []
    counter = 0
    with open(PATH, 'r') as f:
        for l in f:
            if l[0].isdigit(): # an id line
                if len(codes) != 0:
                    # print (id)
                    # print (codes)
                    # input()
                    counter += 1
                    for code in codes:
                        code2id[normalize_icd9(code)] = id
                    codes = []
                id = l.split()[0]
            elif l[0] == ' ': # a code line
                vec = l.strip().split()
                codes.extend(vec)
    print ('Loaded %d icd9 groups from ccs' % counter)
    return code2id


def read_medication_name_mapping(PATH):
    code2name = {}
    with open(PATH, 'r') as f:
        for l in f:
            vec = l.strip().split('\t')
            name = 'M_' + '_'.join(vec[1].split())
            code2name[vec[0]] = name
    print ('Loaded %d medication groups from d_medication\n i.e. %s' % (len(code2name), vec[0]+':'+name))
    return code2name


def pretty_print(li):
    for elem in li:
        print(elem)
    print()


def load_reverse_pair(PATH):
    pairs = {}
    with open(PATH, 'r') as f:
        for l in f.readlines():
            if len(l.strip()) > 0:
                vec = l.strip().split('\t')
                pairs[vec[1]] = vec[0]  # 2 -> 1
    return pairs


def load_pair(PATH):
    pairs = {}
    with open(PATH, 'r') as f:
        for l in f.readlines():
            if len(l.strip()) > 0:
                vec = l.strip().split('\t')
                pairs[vec[0]] = vec[1]  # 1 -> 2
    return pairs


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pkl.dump(obj, MacOSFile(f), protocol=pkl.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pkl.load(MacOSFile(f))



if __name__ == '__main__':
    icd10icd9_path = '/Users/yxiang1/PycharmProjects/asthma/data/corpus/exacerbation/dict/2015_I10gem.txt'
    # icd10icd9_dict = read_icd10icd9_mapping(icd10icd9_path)
    ccs_multi_path = '/Users/yxiang1/PycharmProjects/asthma/data/corpus/exacerbation/dict/ccs_multi.dict'
    # code2id = read_ccs_icd9_multi_mapping(ccs_multi_path)
    # print (len(code2id))
    # print (icd10icd9_dict)
    ndc2name_path = '/Users/yxiang1/PycharmProjects/asthma/data/corpus/exacerbation/dict/ndc2name.dict'
    code2name = read_medication_name_mapping(ndc2name_path)
    print (code2name)


