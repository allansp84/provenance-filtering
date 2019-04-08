# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib


matplotlib.use('Agg')


def spatial_consistency_ransac(psrc, pdst, in_ratio=0.5, min_in_number=1):
    """ Performs spatial consistent of two sets of matching points by estimating a Fundamental Matrix using RANSAC.

    Fundamental matrix estimation is done using OpenCV's estimator.

    :param psrc: first set (source) of matching points;
    :param pdst: second set (destiny) of matching points;
    :param in_ratio: minimum ratio of inliers to confirm consistency;
    :param min_in_number: minimum number of inliers to confirm consistency. Checked together with ratio;

    :return: boolean confirming or deconfirming consistency; array of booleans indicating which matching pairs of points
             are inliers;
    """

    assert psrc.shape[0] == pdst.shape[0]

    total_m_pts = psrc.shape[0]

    pr1 = 3
    pr2 = 0.99
    # print("pr: ({0:0.1f}, {1:0.2f})".format(pr1, pr2))
    funmat, inl = cv2.findFundamentalMat(psrc.astype(np.float32),
                                         pdst.astype(np.float32),
                                         method=cv2.FM_RANSAC,
                                         param1=pr1,
                                         param2=pr2)

    in_num = inl.sum()

    np.set_printoptions(precision=5, suppress=True)

    # print("        > rep_error: ", rep_error)
    # print("        > total_m_pts: ", total_m_pts)
    # print("        > in_num: ", in_num)
    # print("        > ratio: ", float(in_num)/float(total_m_pts))

    if in_num > min_in_number and (float(in_num) / float(total_m_pts) >= in_ratio):
        return True, inl

    else:
        return False, inl


def _line_attributes(psrc, pdst):
    """ Computes line attributes between matching points.

    The line attributes are: lenghts and angles. Notice that the destiny points have to be translated horizontally using
    the width of the source image, as we consider that the lines that link the source points from the source image with
    the destiny points from the destiny image are computed as if both images were displayed side by side, top aligned.
    Angles are in range of 0-360.

    :param psrc: matching points in the source image;
    :param pdst: matching points in the destiny image, already translated horizontally using the source image's width;

    :return: array of line lenghts; array of line angles;
    """

    llengths = np.linalg.norm(psrc - pdst, axis=1)

    # print("psrc, pdst, ll\n", np.vstack([psrc.reshape(-1, 2), pdst.reshape(-1, 2), llengths]))

    diffs = pdst - psrc

    langles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
    langles += 180.0  # just putting the angle in the range 0-360

    return llengths.reshape(-1, 1), langles.reshape(-1, 1)


def _get_line_lenghts(psrc, pdst):
    """ Computes line lenghts between matching points.

    The line attributes are: lenghts and angles. Notice that the destiny points have to be translated horizontally using
    the width of the source image, as we consider that the lines that link the source points from the source image with
    the destiny points from the destiny image are computed as if both images were displayed side by side, top aligned.
    Angles are in range of 0-360.

    :param psrc: matching points in the source image;
    :param pdst: matching points in the destiny image, already translated horizontally using the source image's width;

    :return: array of line lenghts;
    """

    llengths = np.linalg.norm(psrc - pdst, axis=1)

    return llengths.reshape(-1, 1)


def _get_line_angles(psrc, pdst):
    """ Computes line angles between matching points.

    The line attributes are: lenghts and angles. Notice that the destiny points have to be translated horizontally using
    the width of the source image, as we consider that the lines that link the source points from the source image with
    the destiny points from the destiny image are computed as if both images were displayed side by side, top aligned.
    Angles are in range of 0-360.

    :param psrc: matching points in the source image;
    :param pdst: matching points in the destiny image, already translated horizontally using the source image's width;

    :return: array of line angles;
    """

    diffs = pdst - psrc
    langles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
    langles += 180.0  # just putting the angle in the range 0-360

    return langles.reshape(-1, 1)


def line_geom_consistency(psrc, pdst, src_width, f=2.):
    """ Computes the line geometry consistency between two sets of matching points.

    Given two sets of matching points, the lines that link them, along with their attributes (lenght and angle) are
    computed. Assuming there is a certain consistency to the lenghts and angles, we can eliminate matches that deviate
    to much from the mean angle and mean lenght. This deviation is computed as:

    tt = M_a + f*S_a
    tb = M_a - f*S_a

    where M_a is the mean of the attribute (lenght or angle) and S_a its mean deviation. f is an input multiplying
    factor of the number of deviations desired, with default 2.0.

    Any match that has either length or angle more than the corresponding tt or less than the corresponding tb is
    eliminated.

    :param psrc: source image matching points;
    :param pdst: destiny image matching points;
    :param src_width: width of the source image. Used to translate the matching points, so the lines linking the matches
                    can be computed;
    :param f: number of deviations to consider a match inlier or outlier in relation to its lenght or angle;

    :return: array of booleans with the consistency of matches. If position i = True, then the match between psrc[i]
             and pdst[i] is consistent; Line lenghts; Line angles;
    """

    pdst_d = np.array(pdst)
    pdst_d[:, 0] += src_width

    # print("src_width: ", src_width)

    # Get line attributes: lenght and angle.
    ll, la = _line_attributes(psrc, pdst_d)

    llm = np.mean(ll)
    lls = np.std(ll)

    lam = np.mean(la)
    las = np.std(la)

    ll_cons = np.logical_and(ll <= llm + f * lls, ll >= llm - f * lls)
    la_cons = np.logical_and(la <= lam + f * las, la >= lam - f * las)

    cons = np.logical_and(ll_cons, la_cons).reshape(-1)

    return cons, ll, la


def line_geom_consistency_draw(psrc, pdst, dst_disp=100, f=2.):
    """ Supporting function. Does the same as as line_geom_consistnecy, but prints a lot of optional info
        and draws the lenghts and angles distribution.

    :param psrc:
    :param pdst:
    :param dst_disp:
    :param f:
    :return:
    """

    pdst_d = np.array(pdst)
    pdst_d[:, 0] += dst_disp

    # Get line attributes: lenght and angle.
    ll, la = _line_attributes(psrc, pdst_d)

    # print("Line Lenghts: ", ll.reshape(-1, 1), "\n")
    # print("Line Angles: ", la.reshape(-1, 1), "\n")

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title("Histogram of \nMatch Line Lenghts")
    ll_hist, ll_bins, _ = plt.hist(ll, bins=10, color='blue', histtype='bar')
    ax1.set_xlabel('Lenghts')
    ax1.set_ylabel('Count')

    ax2 = fig.add_subplot(122)
    ax2.set_title("Histogram of \nMatch Line Angles")
    la_hist, la_bins, _ = plt.hist(la, bins=360 / 20, range=(0, 360), color='red', histtype='bar')
    ax2.set_xlabel('Angles')
    ax2.set_ylabel('Count')

    plt.tight_layout()
    ax1.plot()
    ax2.plot()

    fig.savefig('histograms.png')

    # fig = plt.figure()
    # la_hist, la_bins, _ = plt.hist(la, bins=360/10, range=(0, 360), color='red', histtype='bar')
    # plt.plot()
    # fig.savefig('line_angles_hist.png')

    llm = np.mean(ll)
    lls = np.std(ll)

    lam = np.mean(la)
    las = np.std(la)

    print("ll_hist: ", ll_hist)
    print("ll_mean: ", llm)
    print("ll_std: ", lls, "\n")

    print("la_hist: ", la_hist)
    print("la_mean: ", lam)
    print("la_std: ", las, "\n")

    ll_cons = np.logical_and(ll <= llm + f * lls, ll >= llm - f * lls)
    la_cons = np.logical_and(la <= lam + f * las, la >= lam - f * las)

    return np.logical_and(ll_cons, la_cons)
