# -*- coding: utf-8 -*-

import sys
import cv2
import time
import numpy as np

hessianThreshold = 100
nOctaves = 5
nOctaveLayers = 5
extended = False
upright = False


def local_feature_detection(img, detetype, kmax=500, mask=None, default_params=True):
    """ Sparsely detects local detection in an image.

    OpenCV implementation of various detectors.

    :param mask:
    :param default_params:
    :param img: input image;
    :param detetype: type of detector {SURF, SIFT, ORB, BRISK}.
    :param kmax: maximum number of keypoints to return. The kmax keypoints with largest response are returned;

    :return: detected keypoins; detection time;
    """

    try:
        if detetype == "SURF":
            if default_params:
                surf = cv2.xfeatures2d.SURF_create()
            else:
                print("SURF: hessianThreshold = {0}".format(hessianThreshold), flush=True)
                print("SURF: nOctaves = {0}".format(nOctaves), flush=True)
                print("SURF: nOctaveLayers = {0}".format(nOctaveLayers), flush=True)

                surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves,
                                                   nOctaveLayers=nOctaveLayers, extended=extended, upright=upright)

            st_t = time.time()
            keypoints = surf.detect(img, mask)
            ed_t = time.time()

            step_size = 5
            if len(keypoints) < kmax:
                keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
                r_state = np.random.RandomState(7)
                keypoints = list(r_state.permutation(keypoints))

            if kmax != -1:
                keypoints = keypoints[0:kmax]

        else:
            ed_t, st_t = 0, 0
            keypoints = []

        det_t = ed_t - st_t
        return keypoints, det_t

    except Exception:
        sys.stderr.write("Failure in detecting detection\n")
        return [], -1


def local_feature_description(img, keypoints, desctype, default_params=True):
    """ Describes the given keypoints of an image.

    OpenCV implementation of various descriptors.

    :param default_params:
    :param img: input image;
    :param keypoints: computed keypoints;
    :param desctype: type of descriptor {SURF, SIFT, ORB, BRISK, RootSIFT}.

    :return: computed detection, description time.
    """

    try:
        if desctype == "SURF":
            if default_params:
                surf = cv2.xfeatures2d.SURF_create()
            else:
                print("SURF: hessianThreshold = {0}".format(hessianThreshold), flush=True)
                print("SURF: nOctaves = {0}".format(nOctaves), flush=True)
                print("SURF: nOctaveLayers = {0}".format(nOctaveLayers), flush=True)

                surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves,
                                                   nOctaveLayers=nOctaveLayers, extended=extended, upright=upright)

            st_t = time.time()
            __, features = surf.compute(img, keypoints)
            ed_t = time.time()

        else:
            ed_t, st_t = 0, 0
            features = []

        dsc_t = ed_t - st_t
        return features, dsc_t

    except Exception:
        sys.stderr.write("Failure in detecting detection\n")
        return [], -1


def local_feature_detection_and_description(imgpath, detetype, desctype, kmax=500, img=None, mask=None, default_params=True):
    """ Given a path or an image, detects and describes local detection.

    :param default_params:
    :param mask:
    :param imgpath: path to the image
    :param detetype: type of detector {SURF, SIFT, ORB, BRISK}.
    :param desctype: type of descriptor {SURF, SIFT, ORB, BRISK, RootSIFT}.
    :param kmax: maximum number of keypoints to return. The kmax keypoints with largest response are returned;
    :param img: (optional) input image. If not present, loads the image from imgpath.

    :return: detected keypoints, described detection, detection time, description time.
    """
    if img is None:
        img = cv2.imread(imgpath)

    try:
        keyps, det_t = local_feature_detection(img, detetype, kmax, mask, default_params)
        if not keyps:
            raise ValueError

        feat, dsc_t = local_feature_description(img, keyps, desctype, default_params)
        if feat is []:
            raise ValueError

        return keyps, feat, det_t, dsc_t

    except ValueError:
        return [], [], -1, -1
