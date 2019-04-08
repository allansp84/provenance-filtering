# -*- coding: utf-8 -*-

import sys
# import hdidx

from provenancefiltering.icip17.utils import *

itype_map = {"PQ": 0,
             }

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


def create_lb_index(features, itype, params, niter=100):
    try:
        flann_index = None

        if itype == "PQ":
            # flann_index = hdidx.indexer.PQIndexer()
            # flann_index.build({'vals': features, 'nsubq': params})
            # flann_index.add(features)
            pass

        elif itype == "IVFPQ" or itype == "RIVFPQ":
            # flann_index = hdidx.indexer.IVFPQIndexer()
            # flann_index.build({'vals': features, 'nsubq': params})
            # flann_index.add(features)
            pass

        else:
            pass

        return flann_index, None

    except Exception:
        sys.stderr.write("Could not create index!\n")

        return None


def load_lb_index(features, idxpath):
    try:

        flann_index = load_object(idxpath)

    except Exception:
        raise (Exception, "Could not load index!")

    return flann_index


def save_lb_index(flann_index, outpath):
    try:
        save_object(flann_index, outpath)

        return True

    except Exception:
        sys.stderr.write("Could not save index!\n")

        return False
