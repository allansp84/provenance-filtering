# -*- coding: utf-8 -*-

import os
import errno
import configparser
import numpy as np


def safe_create_dir(dirname):
    """ Safely creates dir, checking if it already exists.

    Creates any parent directory necessary. Raises exception
    if it fails to create dir for any reason other than the
    directory already existing.

    :param dirname: of the directory to be created
    """

    try:
        os.makedirs(dirname)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def map_config(cfgfile):
    """ Map the parameters of a config file to a dictionary of configurations.

    :param cfgfile: path to the configuration file;

    :return: mapping of configurations.
    """

    cfgparser = configparser.ConfigParser()
    cfgparser.read(cfgfile)

    section_maps = {}

    for section in cfgparser.sections():

        options = cfgparser.options(section)
        if section == "Values":
            for opt in options:
                section_maps[opt] = cfgparser.getint(section, opt)

        else:
            for opt in options:
                section_maps[opt] = cfgparser.get(section, opt)

    return section_maps


def keypoints_to_points(keyps):
    """ Given a list of keypoints in OpenCV's format, returns an with the xy coordinates of
        each keypoint.

    :param keyps: keypoints to be converted.

    """

    ptlist = []
    for kp in keyps:
        ptlist.append(kp.pt)

    return np.array(ptlist, dtype=np.float32)


def load_feature_indexing(findexingpath):
    """ Loads a feature indexing file. This files relates the detection from a set to their source
        images, through indexing.

    Each row of this file contains three information:
    <img> <nfeat> <topi>

    Where <img> is the name of the source image, nfeat is the number of detection extracted from <img> and
    <topi> is the index of the top feature that do not belong to that image. To compute the range of detection
    from <img>, we start from index <topi> - <nfeat>, until <topi>. For example, if we have:

    image_X 455 1512

    it means that image_X has 455 detection, in the range of indexes [1512-455 1511] or [1057 1511]

    Note: this format could be simplified to only use the topi, since we know detection start indexing from 0.

    :param findexingpath: path to the feature indexing file.

    :return: numpy array containing the namelist of the database;
    :return: numpy array indexing detection to image indextable[i] = X means that feature[i] belongs to
                         image X;
    :return: numpy array containing the number of detection of each image;
    :return: numpy array containing the top indexes for each image;

    """

    try:

        idxdt = dict(names=('name', 'nfeat', 'topidx'), formats=('S100', np.int32, np.int32))
        dbfeatindex = np.genfromtxt(findexingpath, dtype=idxdt)
        nametable = np.array(dbfeatindex['name'])
        featntable = np.array(dbfeatindex['nfeat'])
        topidxtable = np.array(dbfeatindex['topidx'])

        nametable = np.array([os.path.splitext(nm)[0] for nm in nametable])

        indextable = []

        for idx, n in enumerate(featntable):

            indextable += n*[idx]

        indextable = np.array(indextable, dtype=np.int32)

        return nametable, indextable, featntable, topidxtable

    except Exception:
        return None, None, None, None
