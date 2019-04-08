# /usr/bin/env python
# -*- coding: utf-8 -*-

"""
objret.search.pfidx.py

FLANN index construction and configuration. FLANN indexes are used to perform fast approximate queries over large
sets of elements. The functions in this file use te Native FLANN index - pyflann.

"""

import sys
import pyflann

from provenancefiltering.icip17.utils import total_time_elapsed, get_time

itype_map = {"LINEAR": 0,
             "KDFOREST": 1,
             "KMEANS": 2,
             "COMPOSITE": 3,
             "KDFOREST_SINGLE": 4,
             "KDFOREST_CUDA": 7,
             "HCAL": 5,
             "LSH": 6,
             "SAVED": 254,
             "AUTO": 255}

dtype_map = {"EUCLIDEAN": "euclidean",
             "L2": "euclidean",
             "MANHATTAN": "manhattan",
             "L1": "manhattan",
             "CHISQUARE": "cs",
             "CSQ": "cs",
             "KULLBACK-LEIBLER": "kl",
             "KBL": "kl",
             "HELLINGER": "hellinger",
             "HLG": "hellinger",
             "HAMMING": "hamming",
             "HMD": "hamming"}


def create_pf_flann_index(features, itype, dtype, n_kdtree_kpm):
    """ Creates a flann index using the FLANN library.

    :param n_kdtree_kpm:
    :param features: array of detection to build the index for
    :param itype: type of index {"LINEAR", "KDFOREST", "KMEANS", "COMPOSITE", "KDFOREST_SINGLE", "HCAL", "LSH", "SAVED",
             "AUTO"};
    :param dtype: type of distance {"EUCLIDEAN" or "L2", "MANHATTAN" or "L1", "CHISQUARE" or "CSQ",
                  "KULLBACK-LEIBLER" or "KBL", "HAMMING" or "HMD", "HELLINGER" or "HEL"};

    :return: FLANN index, index building parameters;
    """

    try:
        if itype == "LSH":

            flann_index = pyflann.FLANN()
            # # import pdb; pdb.set_trace()
            # import numpy as np
            # feat_min = detection.min(axis=1).reshape((-1, 1))
            # feat_max = detection.max(axis=1).reshape((-1, 1))
            # feat_div = feat_max - feat_min
            # feat_div[feat_div == 0.0] = 1
            # detection = 100*(detection - feat_min)/feat_div
            idx_params = flann_index.build_index(features, algorithm="lsh", table_number_=12, key_size_=20, multi_probe_level_=0)

        elif itype == "HCAL":

            flann_index = pyflann.FLANN()

            idx_params = flann_index.build_index(features,
                                                 algorithm="hierarchical",
                                                 branching=16,
                                                 trees=n_kdtree_kpm,
                                                 leaf_max_size=10)

        elif itype == "KDFOREST":

            pyflann.set_distance_type(dtype_map[dtype])

            flann_index = pyflann.FLANN()

            idx_params = flann_index.build_index(features,
                                                 algorithm="kdtree",
                                                 trees=n_kdtree_kpm,
                                                 random_seed=24,
                                                 )

        elif itype == "KDFOREST_SINGLE":

            pyflann.set_distance_type(dtype_map[dtype])

            flann_index = pyflann.FLANN()

            idx_params = flann_index.build_index(features,
                                                 algorithm="kdtree_single",
                                                 leaf_max_size=16)

        elif itype == "KDFOREST_CUDA":

            pyflann.set_distance_type(dtype_map[dtype])

            flann_index = pyflann.FLANN()

            idx_params = flann_index.build_index(features,
                                                 algorithm="kdtree",
                                                 trees=10)

        elif itype == "KMEANS":

            flann_index = pyflann.FLANN()

            pyflann.set_distance_type(dtype_map[dtype])

            idx_params = flann_index.build_index(features,
                                                 algorithm="kmeans",
                                                 branching=32,
                                                 iterations=11,
                                                 centers_init=0,
                                                 cb_index=0.2)

        elif itype == "AUTO":

            # pyflann.set_distance_type(dtype_map[dtype])

            flann_index = pyflann.FLANN()

            idx_params = flann_index.build_index(features,
                                                 algorithm="autotuned",
                                                 target_precision=0.5,
                                                 build_weight=0.00001,
                                                 memory_weight=0,
                                                 sample_fraction=1,
                                                 log_level="info")
            print(idx_params)

        elif itype == "LINEAR":

            # -- pyflann.set_distance_type(dtype_map[dtype])

            flann_index = pyflann.FLANN()

            idx_params = flann_index.build_index(features,
                                                 algorithm="linear")
        else:
            raise Exception("Could not create index!\n")

    except Exception:
        raise Exception("Could not create index!\n")

    return flann_index, idx_params


def load_pf_flann_index(features, idxpath):
    """ Loads a FLANN index, implemented with native FLANN library.

    :param features: detection used to build the index. Must agree with the index;
    :param idxpath: path to the index file;

    :return: FLANN index.
    """

    try:
        print("-- Loading indexing structure")
        start = get_time()

        flann_index = pyflann.FLANN()

        flann_index.load_index(idxpath, features)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    except Exception:
        raise Exception("Could not load index!\n")

    return flann_index


def save_pf_flann_index(flann_index, outpath):
    """ Save a built FLANN index to a file.

    :param flann_index: FLANN index;
    :param outpath: outfile;

    :return: True if successful.
    """

    try:
        flann_index.save_index(outpath)
    except Exception:
        raise Exception("Could not save index!\n")

    return True
