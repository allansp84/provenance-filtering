# -*- coding: utf-8 -*-

import os
import sys
import cv2
import time
import argparse

from provenancefiltering.icip17.controller import Controller
from provenancefiltering.icip17.datasets import registered_datasets
from provenancefiltering.icip17.utils import N_JOBS
from provenancefiltering.icip17.subspaces import subspace_algo


class CommandLineParser(object):

    def __init__(self):
        # -- define the arguments available in the command line execution
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    def parsing(self):
        dataset_options = 'Available dataset interfaces: '
        for k in sorted(registered_datasets.keys()):
            dataset_options += ('%s-%s, ' % (k, registered_datasets[k].__name__))

        subspace_algo_options = 'Available Algorithm for Subspace: '
        for k in sorted(subspace_algo.keys()):
            subspace_algo_options += ('%s-%s, ' % (k, subspace_algo[k]))

        # -- common parameters
        group_a = self.parser.add_argument_group('Arguments')

        group_a.add_argument('--dataset', type=int, metavar='', default=0, choices=range(len(registered_datasets)),
                             help=dataset_options + '(default=%(default)s).')

        group_a.add_argument('--dataset_path', type=str, metavar='', default='',
                             help='Path to the videos of valid accesses and attempted attacks.\n\n')

        group_a.add_argument('--output_path', type=str, metavar='', default='',
                             help='Path where the results will be saved.')

        group_a.add_argument('--groundtruth_path', type=str, metavar='', default='',
                             help='A *.csv file containing the ground-truth.')

        group_a.add_argument('--feature_extraction', action='store_true',
                             help='Extract detection (default=%(default)s).')

        group_a.add_argument('--matching', action='store_true',
                             help='Matching (default=%(default)s).')

        group_a.add_argument('--plot_pr_curves', action='store_true',
                             help='Plot Precision-Recall Curves (default=%(default)s).')

        group_a.add_argument('--mean_curves', action='store_true',
                             help='Plot Mean Precision-Recall Curves (default=%(default)s).')

        group_a.add_argument('--n_jobs', type=int, metavar='int', default=N_JOBS,
                             help='Number of jobs to be used during processing (default=%(default)s).')

        # -- KEY-POINT MATCHING-BASED APPROACHES
        group_b = self.parser.add_argument_group('Arguments Used in the Approaches Based on Key-Point Matching')

        detector_kpm_options = ["SURF", "SIFT", "ORB", "DENSE", "BRISK", "BINBOOST"]
        group_b.add_argument("--detector_kpm",
                             type=str.upper,
                             metavar="",
                             default="SURF",
                             help="Allowed values are: " + ", ".join(
                                 detector_kpm_options) + ' (default=%(default)s).')

        descriptor_kpm_options = ["SURF", "SIFT", "ORB", "RootSIFT", "RootSURF", "BRISK", "BINBOOST"]
        group_b.add_argument("--descriptor_kpm",
                             type=str.upper,
                             metavar="",
                             default="SURF",
                             help="Allowed values are: " + ", ".join(
                                 descriptor_kpm_options) + ' (default=%(default)s).')

        group_b.add_argument("--limit_kpm",
                             type=int,
                             metavar="",
                             default=500,
                             help="Maximum number of keypoints to be detected by"
                                  "sparse detectors (default=%(default)s).",
                             )

        group_b.add_argument("--interval_kpm",
                             type=int,
                             metavar="",
                             nargs=2,
                             default=[-1, -1],
                             help="Pair of number enumerating the images to start and end extraction. Expects \n"
                                  "a start value and an end value. For example, -i 10 20 extracts detection \n"
                                  "from image 10 to image 19 (default=%(default)s).",
                             )

        save_kpm_options = ["individual", "all", "both"]
        group_b.add_argument("--save_kpm",
                             type=str,
                             metavar="",
                             default="both",
                             help="Indicates the files to save. Allowed values are: " +
                                  ", ".join(save_kpm_options) + ". \n"
                                                                "'individual' saves one feature vector file for each image. \n" +
                                  "'all' saves one file for all the detection extracted from multiple images. \n"
                                  "'both' saves both cases. (default=%(default)s)")

        group_b.add_argument("--outtype_kpm",
                             type=int,
                             default=0,
                             choices=[0, 1],
                             help="Type of the output file. 0 for binary files, 1 for txt files (default=%(default)s)."
                             )

        index_type_kpm_options = ["LINEAR", "KMEANS", "KDFOREST", "KDFOREST_SINGLE", "LSH", "HCAL", "AUTO",
                                  "PQ", "IVFPQ", "RIVFPQ", "SH", "MIH"]
        group_b.add_argument("--index_type_kpm",
                             type=str,
                             metavar="",
                             default="LINEAR",
                             help="Allowed values are: " + ", ".join(
                                 index_type_kpm_options) + ' (default=%(default)s).')

        group_b.add_argument("--n_kdtree_kpm",
                             type=int,
                             metavar="",
                             default=10,
                             help="Number of parallel k-d tree to use (default=%(default)s).")

        distance_kpm_optins = ["L1", "MANHATTAN", "L2", "EUCLIDEAN", "HMD", "HAMMING", "CSQ", "CHISQUARE",
                               "KBL", "KULLBACK-LEIBLER", "HEL", "HELLINGER"]
        group_b.add_argument("--distance_kpm",
                             type=str,
                             metavar="",
                             default="HAMMING",
                             help="Type of distance. Allowed values are: " + ", ".join(
                                 distance_kpm_optins) + ' (default=%(default)s).')

        search_type_kpm_options = ["knn", "radius"]
        group_b.add_argument("--search_type_kpm",
                             type=str,
                             metavar="",
                             default="knn",
                             help="Type of search. Allowed values are: " + ", ".join(
                                 search_type_kpm_options) + ' (default=%(default)s).')

        group_b.add_argument("--rfactor_kpm",
                             type=float,
                             metavar="",
                             default=1.0,
                             help="Factor applied to the standard deviation when computing a search radius. \n"
                                  "The search radius is computed as r = M + f*S, where M and S are the mean \n"
                                  "and std. dev. of the k-th nearest neighbor distances of the 10 first \n"
                                  "matches (default=%(default)s).")

        score_type_kpm_options = ["vote", "distance"]
        group_b.add_argument("--score_type_kpm",
                             type=str,
                             metavar="",
                             default="vote",
                             help="Type of score. Allowed values are: " + ", ".join(score_type_kpm_options) +
                                  " (default=%(default)s). \n"
                                  "If 'vote', uses the count of matches between images to order results. \n"
                                  "If 'distance', uses the mean distance to order results.")

        group_b.add_argument("--n_neighbors_kpm",
                             type=int,
                             metavar="",
                             default=10,
                             help="In 'knn', it is the # of neighbors. "
                                  "In 'radius' is the maximum # of results (default=%(default)s).")

        group_b.add_argument("--lib_kpm", "-l",
                             help="Library to use when creating the flann index. \'cv\' uses opencv, while '\pf\' uses FLANN"
                                  " directly. Default is \'pf\' ",
                             type=str,
                             choices=["cv", "pf", "lb", "lopq", "sk"],
                             default="pf")

        group_b.add_argument("--nsubq_kpm",
                             help="(default=%(default)s).",
                             type=int,
                             choices=[1, 8, 16, 32, 64, 128],
                             default=8)

        group_b.add_argument("--filter_matches",
                             help="(default=%(default)s).",
                             type=str,
                             choices=['', 'good', 'bad'],
                             default='')

        group_b.add_argument("--merge_type",
                             help="(default=%(default)s).",
                             type=str,
                             # choices=['sum', 'max', 'min', 'mean', 'prod',
                             #          'normsum', 'normmax', 'normmin', 'normmean', 'normprod', 'pos'],
                             default='sum')

        group_b.add_argument("--force_written",
                             help="(default=%(default)s).",
                             type=int,
                             default=1)

        group_b.add_argument("--niter_pq",
                             help="(default=%(default)s).",
                             type=int,
                             default=100)

        group_b.add_argument("--query_id",
                             help="(default=%(default)s).",
                             type=str,
                             default='')

        group_b.add_argument("--db_filetype",
                             help="(default=%(default)s).",
                             type=str,
                             choices=['npy', 'npz', 'hdf5'],
                             default='npy')

        group_b.add_argument("--default_params",
                             help="(default=%(default)s).",
                             type=bool,
                             default=True)

        group_b.add_argument('--context_based_rerank', action='store_true')

        group_b.add_argument('--use_reference_mask', action='store_true')

        group_b.add_argument('--query_contraction', action='store_true')

        group_b.add_argument('--query_expansion', action='store_true')

        group_b.add_argument('--subsampling', action='store_true')

        group_b.add_argument('--merge_file_ranks', action='store_true')

        group_b.add_argument('--merge_indexing_methods', action='store_true')

        group_b.add_argument('--compute_distances_kpm', action='store_true', help='Matching (default=%(default)s).')

        group_c = self.parser.add_argument_group('Other options')

        group_c.add_argument('--resize_img', action='store_true')

        group_c.add_argument('--use_map', action='store_true')

        group_c.add_argument("--n_batches",
                             type=int,
                             metavar="",
                             default=1,
                             help="(default=%(default)s).")

        group_c.add_argument("--n_round",
                             type=int,
                             metavar="",
                             default=1,
                             help="(default=%(default)s).")

        group_e = self.parser.add_argument_group('Available Parameters for Build the Subspaces')

        group_e.add_argument('--subspace_algo', type=int, metavar='', default=0, choices=range(len(subspace_algo)),
                             help=subspace_algo_options + '(default=%(default)s).')

        group_e.add_argument('--n_components', type=int, metavar='', default=32, help='(default=%(default)s).')

    def get_args(self):
        return self.parser.parse_args()


def call_controller(args):
    print("Running controller")

    dataset = registered_datasets[args.dataset]
    data = dataset(args.dataset_path)
    args.prefix_kpm = str(data.__class__.__name__).lower()

    data.output_path = os.path.join(args.output_path,
                                    str(data.__class__.__name__).lower(),
                                    )

    data.groundtruth_path = args.groundtruth_path

    control = Controller(data, args)
    control.execute_protocol()


def main():
    # -- parsing the command line options
    command_line = CommandLineParser()
    command_line.parsing()

    args = command_line.get_args()
    print('ARGS:', args, flush=True)

    # TODO: To change this hack later using a dictionary of parameters
    if "PQ" in args.index_type_kpm:
        args.n_kdtree_kpm = args.nsubq_kpm

    if "RIVFPQ" in args.index_type_kpm:
        args.niter_pq = 1

    cv2.setNumThreads(args.n_jobs)
    print("-- number of thread used in the opencv library:", cv2.getNumThreads(), flush=True)

    # -- create and execute a Controller object
    call_controller(args)


if __name__ == "__main__":
    start = time.time()

    main()

    elapsed = (time.time() - start)
    print('Total Time Elapsed: {0}.'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed))))
    sys.stdout.flush()
