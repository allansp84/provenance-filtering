# -*- coding: utf-8 -*-

import os
import sys
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt

from abc import ABCMeta
from abc import abstractmethod

from sklearn.metrics import auc
from skimage.util import view_as_windows

from provenancefiltering.icip17.utils import *
from provenancefiltering.icip17.measure.measure import compute_mean_precision_recall_curve


class BaseController(object):
    __metaclass__ = ABCMeta

    def __init__(self, data, args):
        self.data = data
        self.args = args

    @staticmethod
    def write_feat_indexing(filepath, feat_indexing):

        f = open(filepath, 'a')

        imname, featn, feati = feat_indexing[0]
        f.write("{0:<100s} {1:d} {2:d}".format(imname, featn, feati))

        for row in feat_indexing[1:]:
            imname, featn, feati = row
            f.write("\n{0:<100s} {1:d} {2:d}".format(imname, featn, feati))

        f.close()

        return

    @staticmethod
    def find_bounding_box(contour):
        min_x, max_x = contour[:, 0].min(), contour[:, 0].max()
        min_y, max_y = contour[:, 1].min(), contour[:, 1].max()
        width = max_x - min_x
        height = max_y - min_y
        return np.array([min_x, min_y, width, height])

    def show_ground_truth(self, output_path='ground_truth', img_size=256):
        all_fnames = self.data.meta_info['all_fnames']
        all_labels = self.data.meta_info['all_labels']
        all_idxs = self.data.meta_info['all_idxs']
        query_idxs = self.data.meta_info['query_idxs']
        sub_class = self.data.meta_info['sub_class']

        nimg_per_row = 10
        pad_size = 60
        # empty_img = np.zeros((img_size+pad_size, img_size, 3), dtype=np.uint8)
        n_samples = 0

        for q_idx in query_idxs:

            print("-- saving ground truth for query {0}".format(q_idx))
            sys.stdout.flush()

            rel_path = os.path.relpath(all_fnames[q_idx], self.data.dataset_path)
            fname = os.path.join(self.data.dataset_path, rel_path)
            img_0 = cv2.imread(fname, cv2.IMREAD_COLOR)[:, :, ::-1]

            img_0 = cv2.resize(img_0, (img_size, img_size))
            pad = np.zeros(((pad_size,) + img_0.shape[1:]), dtype=np.uint8) + 255

            # img_0 = np.concatenate((img_0, pad), axis=0)
            # cv2.putText(img_0, 'Query', ((10), img_0.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, 1.4,
            #             color=(0,0,0), thickness=2, lineType=cv2.CV_AA)

            imgs = []
            # imgs += [empty_img for i in range(nimg_per_row-1)]

            # fnames = []

            idxs = np.sort(all_idxs[all_labels[q_idx] == 1])
            n_samples += len(idxs)
            for idx in idxs:

                img = cv2.imread(all_fnames[idx], cv2.IMREAD_COLOR)[:, :, ::-1]

                img = cv2.resize(img, (img_size, img_size))
                img = np.concatenate((img, pad), axis=0)

                img_text = "{0}".format(sub_class[idx])
                cv2.putText(img, img_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.4,
                            color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                imgs += [img]

            full_img = mosaic(nimg_per_row, imgs)

            qualitative_result = "{0}/query_{1}.jpg".format(output_path, q_idx)
            safe_create_dir(os.path.dirname(qualitative_result))
            cv2.imwrite(qualitative_result, full_img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        print("Total of samples in the dataset: {0}:".format(n_samples))

    def show_top_k_images_per_query(self, predictions, output_path, top_k=5, n_query=10, img_size=512):

        all_fnames = self.data.meta_info['all_fnames']
        query_idxs = self.data.meta_info['query_idxs']
        sub_class = self.data.meta_info['sub_class']
        pred_idxs = predictions['pred_idxs']
        all_labels = self.data.meta_info['s_all_labels'][0][0]
        host_labels = self.data.meta_info['s_all_labels'][1][0]
        alien_labels = self.data.meta_info['s_all_labels'][2][0]

        # if n_query != -1:
        #     query_idxs = query_idxs[:n_query]
        #     pred_idxs = pred_idxs[:n_query]
        # else:
        rstate = np.random.RandomState(42)
        selected_query = rstate.permutation(query_idxs)[:n_query]
        query_idxs = query_idxs[selected_query]
        pred_idxs = pred_idxs[selected_query]

        nimg_per_row = 5
        pad_size = 60
        empty_img = np.zeros((img_size+pad_size, img_size, 3), dtype=np.uint8)

        print("-- saving qualitative results for queries")
        sys.stdout.flush()

        for q_idx, idxs in zip(query_idxs, pred_idxs):

            # print "-- saving qualitative results for query {0}".format(q_idx)
            # sys.stdout.flush()
            rel_path = os.path.relpath(all_fnames[q_idx], self.data.dataset_path)
            query_fname = os.path.join(self.data.dataset_path, rel_path)
            img_0 = cv2.imread(query_fname, cv2.IMREAD_COLOR)[:, :, ::-1]

            # rate = img_size/float(img_0.shape[0])
            # img_0 = cv2.resize(img_0, (int(img_0.shape[1]*rate), int(img_size)))
            img_0 = cv2.resize(img_0, (img_size, img_size))

            pad = np.zeros(((pad_size,) + img_0.shape[1:]), dtype=np.uint8)+255
            blue_pad = np.zeros(((pad_size,) + img_0.shape[1:]), dtype=np.uint8)
            blue_pad[:, :, 2] = 255

            img_0 = np.concatenate((img_0, pad), axis=0)
            cv2.putText(img_0, 'Query', (10, img_0.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, 1.4,
                        color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            imgs = [img_0]
            imgs += [empty_img for _ in range(nimg_per_row-1)]

            fnames = []
            for t in range(top_k):

                rel_path = "{0}".format(os.path.relpath(all_fnames[idxs[t]], self.data.dataset_path))
                rel_path = rel_path.replace('detection/', '')
                rel_path = rel_path.replace('npy', 'jpg')
                fname = os.path.join(self.data.dataset_path, rel_path)
                fnames += [fname]

                img = cv2.imread(fname, cv2.IMREAD_COLOR)[:, :, ::-1]

                # rate = img_size / float(img.shape[0])
                # img = cv2.resize(img, (int(img.shape[1] * rate), int(img_size)))
                img = cv2.resize(img, (img_size, img_size))

                if all_labels[q_idx, idxs[t]]:
                    img = np.concatenate((img, blue_pad), axis=0)
                else:
                    img = np.concatenate((img, pad), axis=0)

                img_text = "Rank {0} ({1})".format((t+1), sub_class[idxs[t]])

                if len(host_labels):
                    if len(host_labels):
                        if host_labels[q_idx, idxs[t]]:
                            img_text = "{0} (H)".format(img_text)

                if len(alien_labels):
                    if len(alien_labels):
                        if alien_labels[q_idx, idxs[t]]:
                            img_text = "{0} (A)".format(img_text)

                cv2.putText(img, img_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.4,
                            color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                imgs += [img]

            full_img = mosaic(nimg_per_row, imgs)

            qualitative_result = "{0}/qualitative_results/top_{1}/{2}".format(output_path, top_k, os.path.basename(query_fname))
            safe_create_dir(os.path.dirname(qualitative_result))
            cv2.imwrite(qualitative_result, full_img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 65])

            ranked_images = "{0}/ranked_images/top_{1}/{2}.txt".format(output_path, top_k,
                                                                       os.path.splitext(os.path.basename(query_fname))[0])
            safe_create_dir(os.path.dirname(ranked_images))
            np.savetxt(ranked_images, np.array(fnames), fmt="%s")

    def save_ranked_images_per_query(self, predictions, output_path, top_k=5):
        all_fnames = self.data.meta_info['all_fnames']
        all_idxs = self.data.meta_info['all_idxs']
        query_idxs = self.data.meta_info['query_idxs']
        pred_idxs = predictions['pred_idxs']

        for q, idxs in enumerate(pred_idxs):

            print("-- saving qualitative results for query {0}".format(q))
            sys.stdout.flush()

            dir_name = os.path.splitext(os.path.basename(all_fnames[query_idxs[q]]))[0]

            fnames = []
            for t in range(top_k):
                input_fname = all_fnames[idxs[t]]
                rel_path = os.path.relpath(all_fnames[idxs[t]], self.data.dataset_path)
                output_fname = os.path.join(output_path, "ranked_images", dir_name, rel_path)
                safe_create_dir(os.path.dirname(output_fname))
                shutil.copyfile(input_fname, output_fname)

                rel_path = "{0}".format(os.path.relpath(all_fnames[idxs[t]], self.data.dataset_path))
                rel_path = rel_path.replace('detection/', '')
                rel_path = rel_path.replace('npy', 'jpg')
                fname = os.path.join(self.data.dataset_path, rel_path)
                fnames += [fname]

            ranked_images = "{0}/ranked_images/query_{1}_top_{2}.txt".format(output_path, q, top_k)
            safe_create_dir(os.path.dirname(ranked_images))
            np.savetxt(ranked_images, np.array(fnames), fmt="%s")

        rel_fnames = np.array([os.path.relpath(fname, PROJECT_PATH) for fname in all_fnames])
        np.savetxt('relative-filenames-by-indexes.txt',
                   np.concatenate((rel_fnames.reshape(-1, 1), all_idxs.reshape(-1, 1)), axis=1), fmt="%s\t%s")

        rel_fnames = np.array(["{0}_{1}".format(os.path.splitext(os.path.basename(all_fnames[qidx]))[0], qidx) for qidx in query_idxs])
        np.savetxt('query-image-names-by-indexes.txt', np.concatenate((rel_fnames.reshape(-1, 1), query_idxs.reshape(-1, 1)), axis=1),
                   fmt="%s\t%s")
        rel_fnames = np.array(
            ["{0}_{1}".format(os.path.splitext(os.path.basename(all_fnames[idx]))[0], idx) for idx in pred_idxs[0]])

        np.savetxt('image-names-by-indexes.txt',
                   np.concatenate((rel_fnames.reshape(-1, 1), pred_idxs[0].reshape(-1, 1)), axis=1), fmt="%s\t%s")

    def plot_precision_recall_curves(self, mean_curve=False, n_points=100):

        print("Precision-Recall Curves ...")

        descriptor = "Match{0}".format(self.args.descriptor_kpm)
        output_path_graphs = os.path.join(self.data.output_path,
                                          "{0}/{1}/matches".format(descriptor, self.args.limit_kpm))

        fnames = retrieve_samples(output_path_graphs, 'npy')
        fnames = [fname for fname in fnames if 'pr_curve_all.npy' in fname]

        new_r_dict = []
        if mean_curve:
            fnames = np.array(fnames)
            fnames = np.reshape(fnames, (-1, 3)).T

            for fname in fnames:
                recalls, precisions = [], []
                for f in fname:
                    r_dict = load_object(f)
                    curve = r_dict['curve']
                    # title = r_dict['title']

                    recalls += [curve[0, :][::-1]]
                    precisions += [curve[1, :][::-1]]

                r_dict = load_object(fname[0])
                title = r_dict['title']

                mean_recall, mean_precision = compute_mean_precision_recall_curve(recalls, precisions, n_points=n_points)
                area_under_curve_value = auc(mean_recall, mean_precision)
                curve = np.array([mean_recall, mean_precision])
                new_r_dict += [{'curve': curve, 'auc': area_under_curve_value, 'title': title}]

        fig = plt.figure(figsize=(12, 9), dpi=300)
        plt.clf()
        plt.plot([0, 1], [0.5, 0.5], '--', color=(0.6, 0.6, 0.6))

        for i in range(len(fnames)):
            if mean_curve:
                curve = new_r_dict[i]['curve']
                title = new_r_dict[i]['title']
                area_under_curve_value = new_r_dict[i]['auc']

                plot_name = '{0} (AUC={1:0.2f})'.format(title, area_under_curve_value)

            else:
                r_dict = load_object(fnames[i])
                curve = r_dict['curve']
                title = r_dict['title']
                area_under_curve_value = r_dict['auc']
                plot_dirname = os.path.dirname(fnames[i])
                kdtree = plot_dirname.split("_")[-2].upper()

                plot_name = '{0}_{1:02d} (AUC={2:0.2f})'.format(title, int(kdtree), area_under_curve_value)

            plt.plot(curve[0, :], curve[1, :], label=plot_name, marker=MATPLOTLIB_MARKERS[i])

        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.ylim([-0.02, 1.02])
        plt.xlim([-0.02, 1.02])

        title = 'Precision-Recall Curve ({0} dataset)'.format(self.data.__class__.__name__)
        plt.title(title, fontsize=24)
        plt.legend(loc="lower left", prop={'size': 12})

        fig.savefig('{0}/precision-recall-curve.pdf'.format(output_path_graphs))

    def plot_precision_recall_at_ks(self, output_path):

        print("Precision and Recall at Ks ...")

        fnames = retrieve_samples(output_path, 'txt')
        fnames = [fname for fname in fnames if '_output' in fname]

        for i in range(len(fnames)):

            fig = plt.figure(figsize=(10, 6), dpi=300)
            plt.clf()

            curves = np.loadtxt(fnames[i], dtype=np.float32, delimiter=',')

            title = self.args.index_type_kpm
            if self.args.index_type_kpm == 'KDFOREST':
                if self.args.n_kdtree_kpm == 1:
                    title = 'KD-Tree'
                else:
                    title = 'KD-Forest'

            if self.args.index_type_kpm == 'IVFPQ':
                title = "PQ"

            plot_dirname = os.path.basename(fnames[i]).split("_")[0]
            plot_name = '{0}'.format(title)

            plt.plot(curves[:-1, 0], curves[:-1, 1], label='Precision@K', marker=MATPLOTLIB_MARKERS[0], markersize=16)
            plt.plot(curves[:-1, 0], curves[:-1, 2], label='Recall@K', marker=MATPLOTLIB_MARKERS[1], markersize=16)

            plt.xlabel('K', fontsize=30)
            plt.xscale('log')

            plt.ylim([0.0, 1.0])
            plt.yticks(np.arange(0.0, 1.1, 0.2))

            plt.grid(b=True, which='major', color=(0.6, 0.6, 0.6), linestyle='--')
            plt.grid(b=True, which='minor', color=(0.6, 0.6, 0.6), linestyle='--')

            plt.tick_params(axis='both', which='major', labelsize=30)

            plt.title(plot_name, fontsize=36)
            plt.legend(loc="upper center", prop={'size': 28})

            plt.tight_layout()

            fig.savefig('{0}/{1}_{2}_{3}_pr.pdf'.format(output_path, plot_name, plot_dirname, self.data.__class__.__name__))

    @staticmethod
    def plot_rank_curve(scores, frank, top_k=20):

        id_query = "_".join(os.path.basename(frank).split("_")[1:-1])
        normalized_votes = scores[:top_k, 2].astype(np.float32)

        fig = plt.figure(figsize=(15, 12), dpi=300)
        plt.clf()

        ax1 = fig.add_subplot(211)
        x = np.arange(1, top_k + 1)
        ax1.plot(x, normalized_votes, marker=MATPLOTLIB_MARKERS[0])

        ax1.xaxis.set_ticks(np.arange(0, max(x), 5))
        ax1.set_xticks(np.arange(0., top_k + 1, 5))
        ax1.set_xticks(np.arange(0., top_k + 1, 2.5), minor=True)

        ax1.yaxis.set_ticks(np.arange(min(normalized_votes), max(normalized_votes), 1))
        ax1.set_yticks(np.arange(0., 1.01, .2))
        ax1.set_yticks(np.arange(0., 1.01, .05), minor=True)

        ax1.grid(which='minor', alpha=0.3)
        ax1.grid(which='major', alpha=0.7)

        plt.xlabel('Rank Position', fontsize=12)
        plt.ylabel('Votes', fontsize=12)
        title = "Normalized Votes (Query {0})".format(id_query)
        plt.title(title, fontsize=16)

        # -- variance
        stride = 1
        window_size = 10

        windows = view_as_windows(normalized_votes, window_shape=window_size, step=stride)
        variances = []
        for window in windows:
            variances += [np.power(np.std(window, ddof=1), 2.)]

        ax2 = fig.add_subplot(212)
        x = np.arange(len(variances))
        ax2.plot(x, variances, marker=MATPLOTLIB_MARKERS[0])

        ax2.xaxis.set_ticks(np.arange(0, max(x), 5))
        ax2.set_xticks(np.arange(0., len(variances), 5))
        ax2.set_xticks(np.arange(0., len(variances), 2.5), minor=True)

        ax2.yaxis.set_ticks(np.arange(0, max(variances) + 0.01, .01))
        ax2.set_yticks(np.arange(0., max(variances) + 0.01, .01))
        ax2.set_yticks(np.arange(0., max(variances) + 0.01, .005), minor=True)

        plt.xlabel('Rank Position', fontsize=12)
        plt.ylabel('Votes', fontsize=12)
        title = "Variance of the Normalized Votes (Query {0})".format(id_query)
        plt.title(title, fontsize=16)

        # ax.grid(which='both')
        ax2.grid(which='minor', alpha=0.3)
        ax2.grid(which='major', alpha=0.7)

        filename = "{0}.png".format(frank)
        fig.savefig(filename)

    @abstractmethod
    def feature_extraction(self):
        pass

    @abstractmethod
    def classification(self):
        pass

    def execute_protocol(self):

        if self.args.feature_extraction:
            print("Feature extraction ...")
            self.feature_extraction()

        if self.args.matching:
            print("Matching ...")
            self.classification()

        if self.args.plot_pr_curves:
            print("Ploting Precision-Recall Curves ...")
            self.plot_precision_recall_curves(mean_curve=self.args.mean_curves)
