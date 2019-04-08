# -*- coding: utf-8 -*-

import errno
import os
import sys

import numpy as np

from provenancefiltering.icip17.indexing.search.lbidx import create_lb_index
from provenancefiltering.icip17.indexing.search.lbidx import save_lb_index
from provenancefiltering.icip17.indexing.search.pfidx import create_pf_flann_index
from provenancefiltering.icip17.indexing.search.pfidx import save_pf_flann_index

from provenancefiltering.icip17.utils import get_time, total_time_elapsed

verbose = True


def create_outdir(outdir):
    try:
        os.makedirs(outdir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def create_feature_index(featpath, outdir, prefix, itype, dtype, used_lib, params, subsampling, niter, db_filetype='npy'):
    create_outdir(outdir)

    if itype == "LSH" and not (dtype == "HMD" or dtype == "HAMMING"):
        if verbose:
            print("Incompatible index and distance types: ", itype, " x ", dtype)
        pass

    elif not (itype == "LSH" or itype == "LINEAR" or itype == "HCAL") and (dtype == "HMD" or dtype == "HAMMING"):
        if verbose:
            print("Incompatible index and distance types: ", itype, " x ", dtype)
        if verbose:
            print("Changing distance ", dtype, " to L2")
        dtype = "L2"

    if verbose:
        print("Reading detection file")
        sys.stdout.flush()

    try:
        if itype in ["KDFOREST", "PQ", "LOPQ", "IVFPQ", "RIVFPQ"]:
            outpath = "{0:s}/{1:s}_{2:s}_{3:s}_{4}.dat".format(outdir, prefix, itype, dtype, params)
        else:
            outpath = "{0:s}/{1:s}_{2:s}_{3:s}.dat".format(outdir, prefix, itype, dtype)

        if os.path.isfile(outpath):
            print("FLANN Index already exists!")
            sys.stdout.flush()

        else:
            start = get_time()
            dbfeatures = []
            for batch_path in featpath:
                print("-- {0}".format(batch_path))
                sys.stdout.flush()
                if 'npz' in db_filetype:
                    dbfeatures += [np.load(batch_path)['arr_0']]
                else:
                    dbfeatures += [np.load(batch_path)]

            dbfeatures = np.concatenate(dbfeatures)

            elapsed = total_time_elapsed(start, get_time())
            print('-- spent time: {0}!'.format(elapsed))
            sys.stdout.flush()

            if verbose:
                print("Creating FLANN Index: ")
            if verbose:
                print("    > Index Type: ", itype)
            if verbose:
                print("    > Distance type: ", dtype)

            if used_lib == 'pf':
                fidx, idx_params = create_pf_flann_index(dbfeatures, itype, dtype, params)

                if verbose:
                    print(" -- Index Params --")
                    for p in idx_params:
                        print("{0:s}: ".format(p), idx_params[p])

                if verbose:
                    print("Saving index: ", outpath)
                save_pf_flann_index(fidx, outpath)
            elif used_lib == 'lb':
                if subsampling:
                    # -- getting 10% of feature vectors
                    r_state = np.random.RandomState(24)
                    dbfeatures = r_state.permutation(dbfeatures)
                    n_feats = int(len(dbfeatures) * 0.1)
                    dbfeatures = dbfeatures[:n_feats]

                fidx, idx_params = create_lb_index(dbfeatures, itype, params, niter)

                if verbose:
                    print("Saving index: ", outpath)
                save_lb_index(fidx, outpath)

            else:
                pass

    except Exception:
        print("Failure on index creation!")
