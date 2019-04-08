# -*- coding: utf-8 -*-

import sys
import cv2
import errno
import pickle
import datetime
import operator
import resource
import itertools as it

from glob import glob
from multiprocessing import Lock
from multiprocessing import Pool
from multiprocessing import Value

from provenancefiltering.icip17.utils.constants import *

counter = Value('i', 0)
counter_lock = Lock()


class CreateStd(object):
    def __init__(self):
        self.rary = []
        self.gary = []
        self.bary = []
        self.rsum = 0.0
        self.gsum = 0.0
        self.bsum = 0.0
        self.ravgary = []
        self.gavgary = []
        self.bavgary = []


def memory_usage_resource():

    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom *= rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def get_time():
    return datetime.datetime.now()


def total_time_elapsed(start, finish):
    elapsed = finish - start

    total_seconds = int(elapsed.total_seconds())
    total_minutes = int(total_seconds // 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int(round(total_seconds % 60))

    return "{0:02d}+{1:02d}:{2:02d}:{3:02d}".format(elapsed.days, hours, minutes, seconds)


def progressbar(name, current_task, total_task, bar_len=20):
        percent = float(current_task) / total_task

        progress = ""
        for i in range(bar_len):
            if i < int(bar_len * percent):
                progress += "="
            else:
                progress += " "

        print("\r{0}{1}: [{2}] {3}/{4} ({5:.1f}%).{6:30}".format(CONST.OK_GREEN, name, progress, current_task,
                                                                 total_task, percent * 100, CONST.END), end="", flush=True)


def start_process():
    pass


def launch_tasks(arg):

    global counter
    global counter_lock

    index, n_tasks, task = arg

    result = task.run()
    with counter_lock:
        counter.value += 1
        progressbar('-- RunInParallel', counter.value, n_tasks)

    return result


class RunInParallel(object):
    def __init__(self, tasks, n_proc=N_JOBS):

        # -- private attributes
        self.__pool = Pool(initializer=start_process, processes=n_proc)
        self.__tasks = []

        # -- public attributes
        self.tasks = tasks

    @property
    def tasks(self):
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks_list):
        self.__tasks = []
        for t in range(len(tasks_list)):
            self.__tasks.append((t, len(tasks_list), tasks_list[t]))

    def run(self):
        global counter
        counter.value = 0

        # pool_outs = self.__pool.map_async(launch_tasks, self.tasks)
        pool_outs = self.__pool.imap_unordered(launch_tasks, self.tasks)
        self.__pool.close()
        self.__pool.join()

        try:
            # results = [out for out in pool_outs.get() if out]
            results = [out for out in pool_outs if out]
            assert (len(results)) == len(self.tasks)

            print('\n{0}-- finish.{1:30}'.format(CONST.OK_GREEN, CONST.END), flush=True)

        except AssertionError:
            print('\n{0}ERROR: some objects could not be processed!{1:30}\n'.format(CONST.ERROR, CONST.END), flush=True)
            sys.exit(1)


class RunInParallelWithReturn(object):
    def __init__(self, tasks, n_proc=N_JOBS):

        # -- private attributes
        self.__pool = Pool(initializer=start_process, processes=n_proc)
        self.__tasks = []

        # -- public attributes
        self.tasks = tasks

    @property
    def tasks(self):
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks_list):
        self.__tasks = []
        for t in range(len(tasks_list)):
            self.__tasks.append((t, len(tasks_list), tasks_list[t]))

    def run(self):
        global counter
        counter.value = 0

        # pool_outs = self.__pool.map(launch_tasks, self.tasks)
        pool_outs = self.__pool.imap(launch_tasks, self.tasks)
        self.__pool.close()
        self.__pool.join()

        try:
            # results = [out for out in pool_outs.get() if out]
            results = [out for out in pool_outs if out]
            assert (len(results)) == len(self.tasks)

            print('\n{0}-- finish.{1:30}'.format(CONST.OK_GREEN, CONST.END), flush=True)

            return results

        except AssertionError:
            print('\n{0}ERROR: some objects could not be processed!{1:30}\n'.format(CONST.ERROR, CONST.END), flush=True)
            sys.exit(1)


def get_interesting_samples(ground_truth, scores, threshold, n=1, label_neg=-1, label_pos=1):
    """
    Return the n most confusing positive and negative sample indexes. Positive samples have
    scores >= threshold and are labeled label_pos in ground_truth. Negative samples are labeled label_neg.
    @param ground_truth:
    @param scores:
    @param threshold:
    @param n:
    @param label_neg:
    @param label_pos:
    """
    pos_hit = []
    neg_miss = []
    neg_hit = []
    pos_miss = []

    for idx, (gt, score) in enumerate(zip(ground_truth, scores)):
        if score >= threshold:
            if gt == label_pos:
                # -- positive hit
                pos_hit += [idx]
            else:
                # -- negative miss
                neg_miss += [idx]
        else:
            if gt == label_neg:
                # -- negative hit
                neg_hit += [idx]
            else:
                # -- positive miss
                pos_miss += [idx]

    # -- interesting samples
    scores_aux = np.empty(scores.shape, dtype=scores.dtype)

    scores_aux[:] = np.inf
    scores_aux[pos_hit] = scores[pos_hit]
    idx = min(n, len(pos_hit))
    int_pos_hit = scores_aux.argsort()[:idx]

    scores_aux[:] = np.inf
    scores_aux[neg_miss] = scores[neg_miss]
    idx = min(n, len(neg_miss))
    int_neg_miss = scores_aux.argsort()[:idx]

    scores_aux[:] = -np.inf
    scores_aux[neg_hit] = scores[neg_hit]
    idx = min(n, len(neg_hit))
    if idx == 0:
        idx = -len(scores_aux)
    int_neg_hit = scores_aux.argsort()[-idx:]

    scores_aux[:] = -np.inf
    scores_aux[pos_miss] = scores[pos_miss]
    idx = min(n, len(pos_miss))
    if idx == 0:
        idx = -len(scores_aux)
    int_pos_miss = scores_aux.argsort()[-idx:]

    r_dict = {'true_positive': int_pos_hit,
              'false_negative': int_neg_miss,
              'true_negative': int_neg_hit,
              'false_positive': int_pos_miss,
              }

    return r_dict


def save_interesting_samples(int_dict):

    print('-- saving interesting samples', flush=True)

    for key_preds in int_dict.keys():
        for int_key in int_dict[key_preds].keys():
            input_fnames = int_dict[key_preds][int_key]['input_fnames']
            output_fnames = int_dict[key_preds][int_key]['output_fnames']

            # -- save files
            for in_fname, out_fname in zip(input_fnames, output_fnames):

                try:
                    os.makedirs(os.path.dirname(out_fname))
                except OSError:
                    pass

                img = cv2.imread(in_fname, cv2.IMREAD_ANYCOLOR)
                cv2.imwrite(out_fname, img)


def retrieve_samples(input_path, file_type):

    dir_names = []
    for root, subFolders, files in os.walk(input_path):
        for f in files:
            if f[-len(file_type):] == file_type:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)

    fnames = []
    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(input_path, dir_name, '*.' + file_type)))
        fnames += dir_fnames

    return fnames


def __grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return it.zip_longest(fillvalue=fillvalue, *args)


def mosaic(w, imgs):
    """
    Make a grid from images.
    w    -- number of grid columns
    imgs -- images (must have same size and format)
    :param imgs:
    :param w:
    """
    imgs = iter(imgs)
    img0 = imgs.__next__()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = __grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))


def __replace_from_list(string_list, old_str, new_str):
    return map(lambda x: str.replace(x, old_str, new_str), string_list)


def creating_csv(f_results, output_path, test_set, measure):

    f_measure = [f for f in f_results if((test_set in f) and (measure in f))]

    configs, values = [], []
    for f_m in f_measure:
        configs += [os.path.dirname(os.path.relpath(f_m, output_path))]
        values += [float(open(f_m, 'r').readline())]

    configs = __replace_from_list(configs, '/', ',')
    # configs = __replace_from_list(configs, test_set, '')
    # configs = __replace_from_list(configs, 'classifiers,', '')

    reverse = False if 'hter' in measure else True

    results = sorted(zip(configs, values), key=operator.itemgetter(1), reverse=reverse)

    fname = "{0}/{1}.{2}.csv".format(output_path, test_set, measure)
    f_csv = open(fname, 'w')
    f_csv.write("N,LGF,M,CS,SDD,DS,CP,C,%s\n" % str(measure).upper())
    for r in results:
        f_csv.write("%s,%s\n" % (r[0], r[1]))
    f_csv.close()

    print(fname, results[:4], flush=True)


def save_object(obj, fname):

    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass

    fo = open(fname, 'wb')
    pickle.dump(obj, fo)
    fo.close()


def load_object(fname):
    fo = open(fname, 'rb')
    obj = pickle.load(fo)
    fo.close()

    return obj


def safe_create_dir(dirname):
    try:
        os.makedirs(dirname)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
