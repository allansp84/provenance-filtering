# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

mpl.use('Agg')

color_dict = dict(red=(0, 0, 255),
                  blue=(255, 0, 0),
                  green=(0, 255, 0),
                  purple=(180, 0, 180),
                  yellow=(0, 255, 255),
                  black=(0, 0, 0),
                  white=(255, 255, 255),
                  orange=(0, 165, 255),
                  cyan=(255, 255, 0),
                  mint=(201, 252, 189),
                  skyblue=(235, 206, 135),
                  magenta=(255, 0, 255),
                  violet=(150, 62, 255),
                  olivedrab=(58, 238, 179),
                  )

color_seq = ['red',
             'blue',
             'green',
             'purple',
             'yellow',
             'black',
             'white',
             'orange',
             'cyan',
             'mint',
             'skyblue',
             'magenta',
             'violet',
             'olivedrab']

total_colors = len(color_seq)


def draw_matches(im1, keyps1, im2, keyps2, matches, draw_keypoints=True, mcolor=(255, 255, 255), mthickness=1):
    """ Draw matches between a pair of images

    Accepts images either with keypoints already drawn, or without keypoints drawn.

    :param matches:
    :param im1: first matching image;
    :param keyps1: keypoints of image 1;
    :param im2: second matching image;
    :param keyps2: keypoints of image 2;
    :param draw_keypoints: if true, draw the keypoints first;
    :param mcolor: triple with the RGB values of the lines connecting matching keypoints;
    :param mthickness: thickness of the line connecting matching keypoints;

    :return: an image concatenating the two matching images, with lines drawn linking matching keypoins.

    """

    if draw_keypoints:
        im1keyp = cv2.drawKeypoints(im1, keyps1, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        im2keyp = cv2.drawKeypoints(im2, keyps2, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    else:
        im1keyp = im1
        im2keyp = im2

    h1 = im1keyp.shape[0]
    h2 = im2keyp.shape[0]

    if h1 > h2:
        disp_y = h1 - h2
        im2pad = np.pad(im2keyp, ((0, disp_y), (0, 0), (0, 0)), mode='constant', constant_values=0)

        imcat = np.hstack((im1keyp, im2pad))
    elif h2 > h1:
        disp_y = h2 - h1
        im1pad = np.pad(im1keyp, ((0, disp_y), (0, 0), (0, 0)), mode='constant', constant_values=0)

        imcat = np.hstack((im1pad, im2keyp))

    else:
        imcat = np.hstack((im1keyp, im2keyp))

    disp_x = im1keyp.shape[1]

    for ml in matches:

        if ml != []:
            try:
                match = ml[0]
            except (IndexError, TypeError):
                match = ml

            idx1 = match.queryIdx
            idx2 = match.trainIdx

            x1, y1 = keyps1[idx1].pt

            x2, y2 = keyps2[idx2].pt

            x2_disp = x2 + disp_x

            cv2.line(imcat, (int(x1), int(y1)), (int(x2_disp), int(y2)), color=mcolor, thickness=mthickness)

    return imcat


def draw_cluster_matches(im1, keyps1, mpts1, im2, keyps2, mpts2, clusters=None, draw_keypoints=True, mthickness=1):
    """ Draw matches between a pair of images, with optional cluster indexes to differentiate matches
        belonging o different groups

    Accepts images either with keypoints already drawn, or without keypoints drawn.

    :param im1: first matching image;
    :param keyps1: full set of keypoints of image 1. Used only in draw keypoints mode;
    :param mpts1: matched (x, y) points of image 1. Matches are determined by index with mpts2;
    :param im2: second matching image;
    :param keyps2: full set of keypoints of image 2. Used only in draw keypoints mode;
    :param mpts2: matched (x, y) points of image 2. Matches are determined by index with mpts1;
    :param clusters: integers indicating to which cluster the match of index i belong. Accepts a maximum of 14 clusters;
    :param draw_keypoints: if true, draw the keypoints first;
    :param mthickness: thickness of the line connecting matching keypoints;

    :return: an image concatenating the two matching images, with lines drawn linking matching keypoins.
                    matches belonging to different groups are colored differently.

    """

    if clusters is None:
        clusters = np.ones(mpts1.shape[0], dtype=np.int32)
    else:
        assert len(clusters) == mpts1.shape[0]
        assert len(clusters) == mpts2.shape[0]

    if draw_keypoints:
        im1keyp = cv2.drawKeypoints(im1, keyps1, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        im2keyp = cv2.drawKeypoints(im2, keyps2, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    else:
        im1keyp = im1
        im2keyp = im2

    h1 = im1keyp.shape[0]
    h2 = im2keyp.shape[0]

    if h1 > h2:
        disp_y = h1 - h2
        im2pad = np.pad(im2keyp, ((0, disp_y), (0, 0), (0, 0)), mode='constant', constant_values=0)

        imcat = np.hstack((im1keyp, im2pad))
    elif h2 > h1:
        disp_y = h2 - h1
        im1pad = np.pad(im1keyp, ((0, disp_y), (0, 0), (0, 0)), mode='constant', constant_values=0)

        imcat = np.hstack((im1pad, im2keyp))

    else:
        imcat = np.hstack((im1keyp, im2keyp))

    disp_x = im1keyp.shape[1]

    for m1, m2, cl in zip(mpts1, mpts2, clusters):
        x1, y1 = m1

        x2, y2 = m2

        x2_disp = x2 + disp_x

        mcolor = color_dict[color_seq[cl % total_colors]]
        cv2.line(imcat, (int(x1), int(y1)), (int(x2_disp), int(y2)), color=mcolor, thickness=mthickness)

    return imcat


def fancy_dendrogram(*args, **kwargs):
    """ Draws a dendogram. Outside function.
    """

    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = sch.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
