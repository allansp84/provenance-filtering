# -*- coding: utf-8 -*-

import os
import numpy as np
from multiprocessing import cpu_count

# -- available color spaces
color_space_dict = {'grayscale': False, 'rgb': True}

N_JOBS = (cpu_count() - 1) if ((cpu_count()) > 1) else 1
N_JOBS = min(N_JOBS, 3)

DB_FEATS_BATCH_SIZE = 100000

MATPLOTLIB_MARKERS = ["o", "v", "d", "*", "p", "s", "8", "h", "H", "+",
                      "x", "D", "|", "_", ".", ",", "^", "<", ">", "1", "2", "3", "4", ]

PROJECT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
UTILS_PATH = os.path.dirname(__file__)

NC2016_MANIPULATION = os.path.join(os.path.dirname(__file__), "NC2016-manipulation-ref.csv")
NC2016_REMOVAL = os.path.join(os.path.dirname(__file__), "NC2016-removal-ref.csv")
NC2016_SPLICE = os.path.join(os.path.dirname(__file__), "NC2016-splice-ref.csv")
NC2016_FILTERED_SPLICE = os.path.join(os.path.dirname(__file__), "NC2016-filtered-splice-ref.csv")
NIMBLE_ANNOTATION_PATH = os.path.join(os.path.dirname(__file__), "nimble-annotation")
NC2017_Dev1_Beta4 = os.path.join(os.path.dirname(__file__), "NC2017_Dev1_Beta4-splice-ref.csv")
NC2017_Dev1_Beta4_gt = os.path.join(os.path.dirname(__file__), "NC2017_Dev1_Beta4-ground-truth.csv")


class CONST:

    def __init__(self):
        pass

    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


DESCRIPTOR_SIZE = {
    'ORB': 32,
    'BRISK': 64,
    'SIFT': 128,
    'SURF': 64,
    'BINBOOST': 64,
}

DESCRIPTOR_TYPE = {
    'ORB': np.uint8,
    'BRISK': np.uint8,
    'SIFT': np.float32,
    'SURF': np.float32,
    'BINBOOST': np.uint8,
}

WITH_FEAT_INDEX = False

WITH_INDEX = False
