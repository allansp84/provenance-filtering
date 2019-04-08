# -*- coding: utf-8 -*-

import sys
import numpy as np


def count_scores(indextable, votes, votetable, distances, multi=False):
    """ Count vote and distance scores.

    Given an indextable, mapping DB detection to their source images, and a NxK vote array (N being the number
    of detection of the query image and K being the number of neighbors searched), counts the number of votes for
    each image of the DB. An optional NxK distance array can be given as input to compute the average distance
    for the votes of each image.


    :param indextable: table indexing DB detection to their source DB images. indextable[i] = X means that feature i
                       belongs to the X-th image;
    :param votes: NxK array with the KNN of the N query detection;
    :param votetable: Pre-initialized votetable. If empty, it is initialized inside the function. Array counting the
                      number of votes each image has. votetable[i] = k means that the i-th image has k votes;
    :param distances: NxK array with the distances for the KNN of the N query detection;
    :param multi: (optional) Flag indicating if multi-votes can occurs. A multi vote means that a query feature can
                   vote more than once for the same DB image. Defaul is False;
    :return: array counting the number of votes of each DB image, array storing the average vote distance of each DB
             image;
    """

    maxi = indextable[-1]
    if not votetable:
        votetable = [0.0] * (maxi + 1)

    distsum = [0.0] * (maxi + 1)

    # Iterates over the votes (matches).
    r, c = votes.shape
    for ri in range(r):
        voted = []
        vrow = votes[ri, :]

        for ci in range(c):
            voteidx = vrow[ci]

            # indextable indexes a feature number to a DB image.
            imidx = indextable[voteidx]

            if imidx not in voted:

                # Try to get the distance of the corresponding vote.
                try:
                    distval = distances[ri, ci]
                except (NameError, IndexError):
                    distval = 0.0

                votetable[imidx] += 1.0

                distsum[imidx] += distval

                # Do not allow two votes to the same image per query descriptor.
                if not multi:
                    voted.append(imidx)

    try:

        np.seterr(divide='ignore', invalid='ignore')

        votescores = np.array(votetable, dtype=np.float32)

        distscores = np.array(distsum, dtype=np.float32) / votescores
        distscores[votescores == 0] = np.inf

    except Exception:
        sys.stderr.write("Problem creating score tables!\n")
        return None, None

    return votescores, distscores


def normalize_scores(inarray, cvt_sim=False, min_val=-1):
    """ Perform min-max normalization, optionally converting similarity to dissimilarity, and vice versa.

    :param inarray: input array to be normalized;
    :param cvt_sim: If true, converts between similarity-dissimilarity. Conversion is done by subtracting the normalized
                    array from 1;
    :param min_val: (optional) minimum value to use in the min-max normalization. If less than 0, uses the minimum
                    computed from the array.
    :return: normalized array, indices of ordered normalized array.
    """

    if min_val < 0:
        mi = np.min(inarray)
    else:
        mi = min_val

    mx = np.max(inarray)

    norm_array = (inarray.astype(np.float64) - mi) / (mx - mi)
    score_order = norm_array.argsort()

    if cvt_sim:
        return 1 - norm_array, score_order
    else:
        return norm_array, score_order[::-1]
