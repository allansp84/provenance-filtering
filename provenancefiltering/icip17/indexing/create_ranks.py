# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np

from provenancefiltering.icip17.utils import safe_create_dir
from provenancefiltering.icip17.indexing.io import read_array_file
from provenancefiltering.icip17.feature.utility import load_feature_indexing

from provenancefiltering.icip17.indexing.ranking.score import count_scores
from provenancefiltering.icip17.indexing.ranking.score import normalize_scores

verbose = False


def create_rank_files(matchdir,
                      featidxpath,
                      score,
                      limit,
                      force_written):

    if matchdir[-1] != "/":
        matchdir += "/"

    if verbose:
        print("---- Creating rank files ----\n")
    if verbose:
        print("Loading indexing tables: ", featidxpath)
    nametable, indextable, _, _ = load_feature_indexing(featidxpath)
    if nametable is None or indextable is None:
        raise ValueError

    matchflist = glob.glob("{}*.matches".format(matchdir))
    distflist = glob.glob("{}*.dist".format(matchdir))

    matchflist.sort()
    distflist.sort()

    assert len(matchflist) == len(distflist)

    if force_written or not os.path.exists(os.path.join(matchdir, score)):
        for matchfpath, distfpath in zip(matchflist, distflist):

            if verbose:
                print("    > matchfile: ", matchfpath)
            if verbose:
                print("    > distfile: ", distfpath)

            basename = os.path.splitext(os.path.basename(matchfpath))[0]
            basedir = os.path.dirname(matchfpath) + "/"
            rankfpath = "{0:s}/{1:s}/rank_{2:s}.rk".format(basedir, score, basename)

            safe_create_dir(os.path.dirname(rankfpath))

            votes = read_array_file(matchfpath)
            dists = read_array_file(distfpath)

            vtb = [0.0]*len(nametable)
            namearray = np.array(nametable)

            votescores, distscores = count_scores(indextable, votes, vtb, dists, multi=False)

            if votescores is None or distscores is None:
                    raise ValueError

            normvotes, _ = normalize_scores(votescores, cvt_sim=False, min_val=0)

            aux = zip(namearray, votescores, normvotes)

            dt = dict(names=('qname', 'votes', 'normv'),
                      formats=('U100', np.float32, np.float32))

            rank = np.array(list(aux), dtype=dt)

            rank.sort(order=('votes', 'normv', 'qname'))
            rank = rank[::-1]

            if verbose:
                print("    > Writing rank file: ", rankfpath)
            if verbose:
                print("        > ranktype: ", score)

            if limit < 0:
                limit = rank.shape[0]

            np.savetxt(rankfpath, rank[0:limit], fmt="%s,%.5f,%.5f", delimiter=',')

            if verbose:
                print("\n")
    else:
        print("-- Rank files already has been created before!")
        sys.stdout.flush()
