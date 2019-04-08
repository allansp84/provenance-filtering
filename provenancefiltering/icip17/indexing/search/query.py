# -*- coding: utf-8 -*-

import sys
import time


def flann_search(queryfeat, flann_index, k=1, flib="pf"):
    """ Perform FLANN search, specifying which library was used to construct the index.

    Both native FLANN library and OpenCV's interface are available. Because of small differences between both,
    they should be specified by the \'flib\' parameter.

    :param queryfeat: query detection;
    :param flann_index: flann index to search for the DB detection;
    :param k: Number of neighbors;
    :param flib: Flann library used. Either "cv" (OpenCV) or "pf" (Native FLANN);

    :return: array of indices of the matching DB detection, array of distances of the matching DB detection,
             total search time.
    """

    try:
        if flib == "pf":
            ts = time.time()
            indices, dists = flann_index.nn_index(queryfeat,
                                                  num_neighbors=k,
                                                  **dict(checks=32))
            te = time.time()

        elif flib == "lb":
            ts = time.time()
            # result = flann_index.query(queryfeat.tolist(), 2, 10)
            # indices, dists = result[0], result[1]
            params = {'knn': k}
            indices, dists = flann_index.search(queryfeat, k, **params)
            te = time.time()

        else:
            raise Exception("Unrecognized search lib parameter!\n")

        tt = te - ts

        return indices, dists, tt

    except:
        sys.stderr.write("Failure to perform search!\n")
        return [], [], -1.00
