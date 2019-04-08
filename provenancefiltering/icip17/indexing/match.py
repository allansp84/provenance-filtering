# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np

mtype_list = ['Brute', 'Flann']


def match_descriptors(desc_a, desc_b, m_type="Brute", norm_type=cv2.NORM_L2, knn=1, matcher=None):
    """ Matches the given descriptors using OpenCV matchers. KNN matching.

    Two M dimensional sets of N descriptors are given as input, as NxM numpy arrays. Type of matching can be
    specified as either Brute Force or Flann. Distance used is also specified using OpenCV's norm types.
    An input matcher can be given optionally.

    :param desc_a: first set of descriptors;
    :param desc_b: second set of descriptors;
    :param m_type: type of matcher. Either \'Brute\' or \'Flann\';
    :param norm_type: tye of distance. For available types, check cv2.NORM_*. Default is cv2.NORM_L2;
    :param knn: Number of neighbors;
    :param matcher: pre allocated matcher.

    :return: list of lists, each containing K DMatch structures. List of index i contains the list of K
             nearest neighbors in desc_b of the i-th descriptor of desc_a; matching time;

    """

    if not matcher:
        if m_type == "Brute":
            matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
        elif m_type == "Flann":
            matcher = cv2.DescriptorMatcher_create("FlannBased")
        else:
            raise Exception("Unsuported matcher type! {0:s} not in ({1:s})".format(m_type, ", ".join(mtype_list)))

        ts = time.time()
        matched_desc = matcher.knnMatch(desc_a, desc_b, k=knn)
        te = time.time()
    else:
        ts = time.time()
        matched_desc = matcher.knnMatch(desc_a, k=knn)
        te = time.time()

    return matched_desc, te - ts


def get_match_indices(matches):
    """ Get arrays of query and train indexes, given a list of DMatch lists

    :param matches: list of DMatch lists;

    :return: array of query indexes; array of train indexes;

    """

    query_idx = []
    train_idx = []

    for mlist in matches:

        for m in mlist:
            query_idx.append(m.queryIdx)
            train_idx.append(m.trainIdx)

    query_idx = np.array(query_idx, dtype=np.int32)
    train_idx = np.array(train_idx, dtype=np.int32)

    return query_idx, train_idx


def get_match_attributes(matches):
    """ Get arrays of query indexes, train indexes, and distances, given a list of DMatch lists

    Returns the distances additionally

    :param matches: list of DMatch lists;

    :returns query_idx: array of query indexes, array of train indexes, array of distances;

    """

    query_idx = []
    train_idx = []
    dists = []

    for mlist in matches:

        for m in mlist:
            query_idx.append(m.queryIdx)
            train_idx.append(m.trainIdx)
            dists.append(m.distance)

    query_idx = np.array(query_idx, dtype=np.int32)
    train_idx = np.array(train_idx, dtype=np.int32)
    dists = np.array(dists, dtype=np.float32)

    return query_idx, train_idx, dists


def get_matched_keypoints(matches, qkeyp, tkeyp):
    """ Given a list of DMatch lists and two sets of query and train keypoints, returns arrays of matching query
        and train keypoints. Only the xy coordinates are returned.

    :param tkeyp:
    :param qkeyp:
    :param matches: list of DMatch lists;

    :returns q_mkeyp_a: array of query matching keypoints (xy coordinates); array of query matching keypoints
                       (xy coordinates); array of distances.

    """

    q_mkeyp = []
    t_mkeyp = []
    dists = []

    for mlist in matches:

        for m in mlist:
            qi = m.queryIdx
            ti = m.trainIdx

            q_mkeyp.append(qkeyp[qi].pt)
            t_mkeyp.append(tkeyp[ti].pt)
            dists.append(m.distance)

    q_mkeyp_a = np.array(q_mkeyp, dtype=np.float32)
    t_mkeyp_a = np.array(t_mkeyp, dtype=np.float32)
    dists = np.array(dists, dtype=np.float32)

    return q_mkeyp_a, t_mkeyp_a, dists


def get_matched_keypoints_from_list(matches, qkeyp, tkeyp):
    """ Given a list of DMatch lists and two sets of query and train keypoints, returns arrays of matching query
        and train keypoints. Only the xy coordinates are returned.

    :param tkeyp:
    :param qkeyp:
    :param matches: list of DMatch lists;

    :returns q_mkeyp_a: array of query matching keypoints (xy coordinates); array of query matching keypoints
                       (xy coordinates); array of distances.

    """

    q_mkeyp = []
    t_mkeyp = []
    dists = []

    for m in matches:
        qi = m.queryIdx
        ti = m.trainIdx

        q_mkeyp.append(qkeyp[qi].pt)
        t_mkeyp.append(tkeyp[ti].pt)
        dists.append(m.distance)

    q_mkeyp_a = np.array(q_mkeyp, dtype=np.float32)
    t_mkeyp_a = np.array(t_mkeyp, dtype=np.float32)
    dists = np.array(dists, dtype=np.float32)

    return q_mkeyp_a, t_mkeyp_a, dists


def get_matched_points(matches, qpts, tpts):
    """ Given a list of DMatch lists and two sets of query and train points, returns arrays of matching query
        and train keypoints. Only the xy coordinates are returned.

    :param tpts:
    :param qpts:
    :param matches: list of DMatch lists;

    :returns: array of query matching points (xy coordinates); array of query matching points (xy coordinates);
                      array of distances.
    """

    q_mpt = []
    t_mpt = []
    dists = []
    for mlist in matches:

        for m in mlist:
            qi = m.queryIdx
            ti = m.trainIdx

            q_mpt.append(qpts[qi])
            t_mpt.append(tpts[ti])
            dists.append(m.distance)

    q_mpt_a = np.array(q_mpt, dtype=np.float32)
    t_mpt_a = np.array(t_mpt, dtype=np.float32)
    dists = np.array(dists, dtype=np.float32)

    return q_mpt_a, t_mpt_a, dists
