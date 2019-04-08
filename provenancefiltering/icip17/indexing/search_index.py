# -*- coding: utf-8 -*-

import os

from provenancefiltering.icip17.indexing.io import *
from provenancefiltering.icip17.indexing.search.lbidx import load_lb_index
from provenancefiltering.icip17.indexing.search.pfidx import load_pf_flann_index
from provenancefiltering.icip17.indexing.search.query import flann_search
from provenancefiltering.icip17.utils import *

from provenancefiltering.icip17.indexing.ranking.score import *

FV_EXT = [".npy", ".fv", ".bfv"]

verbose = True


def write_times_file(timesfpath, qnamel, timesl):

    mnt = np.mean(timesl)

    tf = open(timesfpath, 'w')

    for qn, t in zip(qnamel, timesl):
        tf.write("{0:<40s} {1:0.5f}\n".format(qn, t))

    tf.write("-----------------\n{0:<40s} {1:0.5f}".format("mean", mnt))

    tf.close()


def search_index(featqueries,
                 dbfeatpath,
                 idxpath,
                 outdir,
                 stype='knn',
                 knn=1,
                 rfactor=1.0,
                 used_lib='pf',
                 force_written=True,
                 db_filetype='npy'):

    if force_written or not os.path.exists(outdir):

        safe_create_dir(outdir)

        if verbose:
            print("Reading DB detection: ", dbfeatpath)
        start = get_time()
        dbfeatures = []
        for batch_path in dbfeatpath:
            if 'npz' in db_filetype:
                dbfeatures += [np.load(batch_path)['arr_0']]
            else:
                dbfeatures += [np.load(batch_path, mmap_mode='r')]
        dbfeatures = np.concatenate(dbfeatures)

        elapsed = total_time_elapsed(start, get_time())
        print('-- spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

        if dbfeatures is None:
            raise ValueError

        flann_index = None
        if verbose:
            print("Loading FLANN index: ", idxpath)

        if used_lib == 'pf':
            flann_index = load_pf_flann_index(dbfeatures, idxpath)

        elif used_lib == 'lb':
            flann_index = load_lb_index(dbfeatures, idxpath)

        elif used_lib == 'lopq':
            flann_index = load_lb_index(dbfeatures, idxpath)
        else:
            pass

        if flann_index is None:
            raise ValueError

        idxbasename = os.path.splitext(os.path.basename(idxpath))[0]
        timesfpath = "{0:s}/{1:s}.times".format(outdir, idxbasename)

        # for i in xrange(len(featqueries)):
        timesl = []
        qnamel = []
        # if os.path.isdir(featqueries[i]):
        #
        #     if verbose: print("Listing query feature vector files.")
        #     if featqueries[i][-1] != "/": featqueries[i] += "/"
        #
        #     featflist = glob.glob(featqueries[i] + '*' + FV_EXT[0])
        #     for ext in FV_EXT[1:]:
        #         featflist.extend(glob.glob(featqueries[i] + '*' + ext))
        #
        # elif os.path.isfile(featqueries[i]) and os.path.splitext(featqueries[i])[1] in FV_EXT:
        #
        #     if verbose: print("Query feature vector file is: ", featqueries[i])
        #     featflist = [featqueries[i]]
        #
        # else:
        #     raise ValueError

        for featfpath in featqueries:

            basename = os.path.splitext(os.path.basename(featfpath))[0]
            matchfpath = "{0:s}/{1:s}.matches".format(outdir, basename)
            distfpath = "{0:s}/{1:s}.dist".format(outdir, basename)

            if force_written or not (os.path.isfile(distfpath) and os.path.isfile(matchfpath)):

                if verbose:
                    print("Reading FV file: {0:s}".format(featfpath))
                if featfpath.endswith('.fv'):
                    qfeat = read_array_file(featfpath)
                else:
                    qfeat = np.load(featfpath)

                if qfeat is not None:
                    if verbose:
                        print("Searching detection on index ({0:s}, {1:d}, {2:0.2f})".format(stype, knn, rfactor))

                    votes, dists, tt = flann_search(qfeat, flann_index, k=knn, flib=used_lib)
                    if verbose:
                        print("Done! ({0:0.2f}s)".format(tt))
                    timesl.append(tt)
                    qnamel.append(basename)

                    if verbose:
                        print("Writing output files: \n")
                    if verbose:
                        print("    > Matches file: ", matchfpath)
                    if verbose:
                        print("    > Distances file:", distfpath, "\n\n")

                    write_array_file(matchfpath, votes)

                    # In case binary detection were searched, convert the distances saved o uint8
                    if qfeat.dtype == np.dtype('uint8'):
                        write_array_file(distfpath, dists.astype('uint8'))
                    else:
                        write_array_file(distfpath, dists)

                else:
                    if verbose:
                        print("No detection could be read from file!")

                if verbose:
                    print("Done searching!\n")

        if timesl:
            if verbose:
                print("Writing times file: ", timesfpath)
            write_times_file(timesfpath, qnamel, timesl)
