# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from provenancefiltering.icip17.utils import N_JOBS
from provenancefiltering.icip17.evaluation.baseevaluation import BaseEvaluation


class Evaluation(BaseEvaluation):

    def __init__(self, output_path, dataset, descriptor, protocol_definition='tvt-protocol',
                 dists_path=None, n_job=N_JOBS):

        super(Evaluation, self).__init__(output_path, dataset, descriptor,
                                         n_job=n_job)

        # private attributes
        self.__output_path = ''

        # public attributes
        self.output_path = output_path
        self.protocol_definition = protocol_definition
        self.debug = True
        self.persist_model = True
        self.dists_path = dists_path

        # self.fusion_type = fusion_type
        self.n_components = 3
        self.decomposition_method = ''
        self.decomposition_model = None
        self.model = None
        self.n_job = n_job

    def load_features(self, fnames):
        dtype = np.uint8
        print('-- loading low level feature ...')
        sys.stdout.flush()

        # n_fnames = len(fnames)
        # feat_dimension = np.load(fnames[0]).shape[1]
        # feats = np.zeros((n_fnames, feat_dimension), dtype=dtype)
        feats = []

        try:

            for i, fname in enumerate(fnames):
                feats += [np.load(fname)[:1, :]]

        except Exception:
            raise Exception('Please, recompute the feature for Video')

        feats = np.array(feats, dtype=dtype)

        return feats

    # @profile
    def get_similarity_matrix(self, query_data=None, query_labels=None, gallery_data=None, gallery_labels=None,
                              train_data=None, train_labels=None):

        print("-- loading similarity matrix", self.dists_path)
        sys.stdout.flush()

        dists_matrix = np.load(self.dists_path)

        # for i in xrange(len(dists_matrix)):
        #     dists_matrix[i, :, 0] = (dists_matrix[i, :, 0] - dists_matrix[i, :, 0].min()) / \
        #                             (dists_matrix[i, :, 0].max() - dists_matrix[i, :, 0].min())

        return dists_matrix

    def get_ranked_lists(self, query_labels=None, gallery_labels=None, all_labels=None, dists_matrix=None):

        ground_truth, all_ranked_labels, pred_idxs, pred_scores = [], [], [], []
        for l, dists in enumerate(dists_matrix):
            scores = dists[:, 0]
            pred_scores += [scores]

            idxs = dists[:, 1].astype(np.int32)
            pred_idxs += [idxs]

            all_ranked_labels += [all_labels[l, idxs]]
            ground_truth += [gallery_labels[l]]

        ground_truth = np.array(ground_truth, dtype=np.uint32)
        all_ranked_labels = np.array(all_ranked_labels, dtype=np.uint32)
        pred_idxs = np.array(pred_idxs, dtype=np.uint32)
        pred_scores = np.array(pred_scores, dtype=np.float32)

        outputs = {'ground_truth': ground_truth,
                   'all_ranked_labels': all_ranked_labels,
                   'pred_idxs': pred_idxs,
                   'pred_scores': pred_scores,
                   }

        return outputs

    def two_fold_protocol(self):
        # -- LOADING THE DATASET
        s_all_labels = self.dataset['s_all_labels']
        gallery_idxs = self.dataset['all_search_idxs']

        # -- starting the training stage
        dists_matrix = self.get_similarity_matrix()

        r_list = {key: [] for _, key in s_all_labels}

        for all_labels, label in s_all_labels:
            print("\n{0} images".format(label))
            sys.stdout.flush()

            # -- starting the testing stage
            r_dict = self.execute_testing(gallery_labels=all_labels[:, gallery_idxs], all_labels=all_labels, dists_matrix=dists_matrix)

            # -- compute MAP and ROC curve
            basename = os.path.basename(self.output_path)
            pr_curve_title = "{0}_{1}_NN{2}_{3}".format(self.descriptor, basename.split('_')[0], basename.split('_')[2], label)
            _ = self.two_fold_performance_evaluation(r_dict, pr_curve_title)

            # -- compute Precision and Recall
            self.compute_top_k(r_dict, label)
            r_list[label] = r_dict

        return r_list
