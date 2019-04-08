# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import itertools

from abc import ABCMeta
from abc import abstractmethod

from sklearn.metrics import auc

from provenancefiltering.icip17.measure.measure import mapk
from provenancefiltering.icip17.measure.measure import recall_threshold
from provenancefiltering.icip17.measure.measure import split_score_distributions
from provenancefiltering.icip17.measure.measure import compute_precision_recall_curve
from provenancefiltering.icip17.measure.measure import compute_mean_precision_recall_curve

from provenancefiltering.icip17.utils import N_JOBS
from provenancefiltering.icip17.utils import load_object
from provenancefiltering.icip17.utils import save_object
from provenancefiltering.icip17.utils import safe_create_dir


class BaseEvaluation(object):
    __metaclass__ = ABCMeta

    def __init__(self, output_path, dataset, descriptor, codebook_size=1000, codebook_selection='kmeans',
                 frame_fusion_type='max', protocol_definition='tvt-protocol', dataset_b=None, output_model=None,
                 n_job=N_JOBS):

        # -- private attributes
        self.__output_path = ''
        self.__frame_numbers = 1
        self.__debug = True

        # -- public attributes
        self.output_path = output_path
        self.output_model = output_model
        self.dataset = dataset
        self.descriptor = descriptor
        self.protocol_definition = protocol_definition
        self.codebook_size = codebook_size
        self.codebook_selection = codebook_selection
        self.frame_fusion_type = frame_fusion_type
        self.dataset_b = dataset_b
        self.n_job = n_job

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        self.__output_path = os.path.abspath(path)

    @abstractmethod
    def load_features(self, fnames):
        return NotImplemented

    @staticmethod
    def load_model(fname, debug=True):

        model = None
        if os.path.isfile(fname):
            model = load_object(fname)
            if debug:
                print('-- found in {0}'.format(fname))
                sys.stdout.flush()
        else:
            if debug:
                print('-- model not found')
                sys.stdout.flush()

        return model

    def plot_precision_recall_curve(self, recall, precision, auc_value, title_legend="Dataset"):

        fig = plt.figure(figsize=(12, 8), dpi=300)

        plt.clf()

        plt.plot([0, 1], [0.5, 0.5], '--', color=(0.6, 0.6, 0.6))

        plt.plot(recall, precision, label=title_legend + ' (AUC={0:0.2f})'.format(auc_value))
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.ylim([0.0, 1.01])
        plt.xlim([0.0, 1.01])
        plt.title('Precision-Recall Curve', fontsize=24)
        plt.legend(loc="lower left")
        # plt.show()
        fig.savefig('{0}/precision-recall_curve.pdf'.format(self.output_path))

    def two_fold_performance_evaluation(self, r_dict, pr_curve_title):

        ranked_labels = r_dict['all_ranked_labels']
        pred_scores = r_dict['pred_scores']

        n_points = 100
        precisions, recalls, aucs = [], [], []
        thresholds = []
        negatives, positives = np.empty((1, 0)), np.empty((1, 0))
        for i, (y_true, scores) in enumerate(zip(ranked_labels, pred_scores)):

            precision, recall, thrs = compute_precision_recall_curve(y_true, scores, pos_label=1, n_points=n_points)

            neg, pos = split_score_distributions(y_true, scores, pos_label=1)
            negatives = np.concatenate((negatives, neg.reshape((1, -1))), axis=1)
            positives = np.concatenate((positives, pos.reshape((1, -1))), axis=1)

            precisions += [precision]
            recalls += [recall]
            thresholds += [thrs]

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        negatives = negatives.flatten()
        positives = positives.flatten()

        mean_recall, mean_precision = compute_mean_precision_recall_curve(recalls, precisions, n_points=n_points)

        mean_auc_pr = auc(mean_recall, mean_precision)

        self.plot_precision_recall_curve(mean_recall, mean_precision, mean_auc_pr)

        curve = np.array([mean_recall, mean_precision])
        file_name = "{0}/pr_curve_{1}.npy".format(self.output_path, str(pr_curve_title.split('_')[-1]).lower())

        output = {'curve': curve,
                  'auc': mean_auc_pr,
                  'title': pr_curve_title
                  }

        save_object(output, file_name)

        recall_values = [0.25, 0.50, 0.75, 1.0]
        recall_thrs = []
        for recall_value in recall_values:
            _, _, recall_thr = recall_threshold(negatives, positives, recall_value, n_points=100)
            recall_thrs += [recall_thr]

        pred_labels_thrs = []
        for recall_thr in recall_thrs:
            pred_labels = np.zeros(pred_scores.shape, dtype=np.int)
            pred_labels[pred_scores >= recall_thr] = 1
            pred_labels_thrs += [pred_labels]

        return output

    def compute_top_k(self, r_dict, label, top=None):

        ground_truth = r_dict['ground_truth']
        all_ranked_labels = r_dict['all_ranked_labels']

        total_positives = np.sum(ground_truth, axis=1).astype(np.float32)

        results_at_k = []

        if top is None:
            top = list(itertools.chain(range(1, 10), range(10, 101, 10), (25,)))

        for n_top in top:

            n_top_ranked_labels = np.sum(all_ranked_labels[:, :n_top], axis=1)
            precision_at_k = n_top_ranked_labels / float(n_top)
            recall_at_k = n_top_ranked_labels / total_positives

            results_at_k += [[n_top, precision_at_k.mean(), recall_at_k.mean(), mapk(all_ranked_labels.tolist(), k=n_top)]]

        output_fname = "{0}/{1}_output.txt".format(self.output_path, str(label).lower())

        safe_create_dir(os.path.dirname(output_fname))

        np.savetxt(output_fname, np.array(results_at_k), fmt="%d,%2.4f,%2.4f,%2.4f", header="k,precision@k,recall@k,map@k")

        print("\nTOP K\t\tMAP@K\t\tPrecision@K\tRecall@K")
        for k, result in enumerate(results_at_k):
            print("{0}\t\t{1:.4f}\t\t{2:.4f}\t\t{3:.4f}".format(result[0],
                                                                mapk(all_ranked_labels.tolist(), k=top[k]),
                                                                result[1], result[2]))

    def execute_testing(self, query_labels=None, gallery_labels=None, all_labels=None, dists_matrix=None):
        r_dict = self.get_ranked_lists(query_labels, gallery_labels, all_labels, dists_matrix)
        return r_dict

    def run(self):
        safe_create_dir(self.output_path)
        r_dict = self.two_fold_protocol()
        return r_dict

    @abstractmethod
    def two_fold_protocol(self):
        return NotImplemented

    @abstractmethod
    def get_similarity_matrix(self, query_data=None, query_labels=None, gallery_data=None, gallery_labels=None,
                              train_data=None, train_labels=None):
        return NotImplemented

    @abstractmethod
    def get_ranked_lists(self, query_labels=None, gallery_labels=None, all_labels=None, dists_matrix=None):
        return NotImplemented
