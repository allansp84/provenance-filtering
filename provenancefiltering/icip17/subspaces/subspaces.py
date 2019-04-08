# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np

from sklearn.decomposition import PCA
from sklearn.utils.extmath import fast_dot

from provenancefiltering.icip17.utils import N_JOBS, progressbar


class Subspaces(object):

    def __init__(self, meta, input_path, output_path,
                 algo=None,
                 output_model=None,
                 descriptor='SURF',
                 n_components=0.95,
                 limit_kpm=500,
                 file_type="npy",
                 n_jobs=N_JOBS):

        # -- private attributes
        self.__input_path = ""
        self.__output_path = ""

        # -- public attributes
        self._meta = meta
        self.input_path = input_path
        self.output_path = output_path
        self.algo = algo
        self.file_type = file_type
        self.n_jobs = n_jobs
        self.seed = 42
        self.debug = True
        self.n_components = n_components
        self.persist = False
        self.limit_kpm = limit_kpm

        self.descriptor = descriptor

        if output_model is None:
            self._fname_subspace_pos = "{0}/subspace/positive_class.subspace".format(self.output_path)
            self._fname_subspace_neg = "{0}/subspace/negative_class.subspace".format(self.output_path)
            self._fname_subspace_unified = "{0}/subspace/unified.subspace".format(self.output_path)

        else:

            self._fname_subspace_pos = "{0}/subspace/positive_class.subspace".format(output_model)
            self._fname_subspace_neg = "{0}/subspace/negative_class.subspace".format(output_model)
            self._fname_subspace_unified = "{0}/subspace/unified.subspace".format(output_model)

    @property
    def input_path(self):
        return self.__input_path

    @input_path.setter
    def input_path(self, path):
        self.__input_path = os.path.abspath(path)

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        path = os.path.abspath(path)
        self.__output_path = path

    @staticmethod
    def __load_features(fnames, n_dimension):

        # feats = np.zeros((0, DESCRIPTOR_SIZE[self.descriptor]), dtype=DESCRIPTOR_TYPE[self.descriptor])
        feats = []
        for i, fname in enumerate(fnames):
            # feats = np.concatenate((feats, np.load(fname)), axis=0)
            feats += [np.load(fname)[:n_dimension, :]]

        feats = np.array(feats)

        # n_feats = feats.shape[0]
        # n_blocksx, n_blocksy = (2, 2)
        # feats = np.reshape(feats, (n_feats, n_blocksx * n_blocksy * 8, -1))

        return feats

    def load_train_features(self):

        if self.debug:
            print('-- loading low level features ...')
            sys.stdout.flush()

        all_fnames = self._meta['all_fnames']
        all_search_idxs = self._meta['all_search_idxs']

        r_state = np.random.RandomState(7)
        sampling_rate = 0.20

        all_search_idxs = r_state.permutation(all_search_idxs)
        n_samples = int(len(all_search_idxs) * sampling_rate)
        train_idxs = all_search_idxs[:n_samples]

        return self.__load_features(all_fnames[train_idxs], n_dimension=500)

    def load_all_features(self):

        if self.debug:
            print('-- loading low level features ...')
            sys.stdout.flush()

        all_fnames = self._meta['all_fnames']

        return self.__load_features(all_fnames, self.limit_kpm), all_fnames

    @staticmethod
    def pickle(fname, data):

        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass

        fo = open(fname, 'wb')
        pickle.dump(data, fo)
        fo.close()

    @staticmethod
    def unpickle(fname):
        fo = open(fname, 'rb')
        data = pickle.load(fo)
        fo.close()
        return data

    def transform(self, subspace, feats):

        if self.debug:
            print('-- coding features ...')
            sys.stdout.flush()

        feats -= subspace['mean']

        transformed_feats = fast_dot(feats, subspace['components'].T)

        # transformed_feats /= np.sqrt(subspace['explained_variance'])

        return transformed_feats

    def save_features(self, feats, fnames):

        print('-- saving transformed features ...')
        sys.stdout.flush()

        for fname, feat_vector in zip(fnames, feats):
            rel_fname = os.path.os.path.relpath(fname, self.input_path)
            output_fname = os.path.join(self.output_path, rel_fname)

            try:
                os.makedirs(os.path.dirname(output_fname))
            except OSError:
                pass

            np.save(output_fname, feat_vector)

    def __subspace_unified(self, feats):
        print('-- fitting a subspace ...')
        sys.stdout.flush()

        if not os.path.exists(self._fname_subspace_unified):
            subspace_algo = {1: PCA(n_components=self.n_components),
                             }

            subspace = subspace_algo[self.algo]

            subspace.fit(np.concatenate(feats))

            self.pickle(self._fname_subspace_unified, subspace)
        else:
            print('-- model already fitted!')
            sys.stdout.flush()

        return True

    def fit_subspace(self):

        if os.path.exists(self._fname_subspace_unified):
            return True

        feats = self.load_train_features()

        self.__subspace_unified(feats)

        return True

    def transform_features(self, feats):

        print('-- feature projection')
        sys.stdout.flush()

        subspace = self.unpickle(self._fname_subspace_unified)

        transformed_feats = []
        for feat in feats:
            transformed_feats += [subspace.transform(feat).astype(np.float32)]

        transformed_feats = np.array(transformed_feats)

        return transformed_feats

    def transform_and_save(self, fnames):
        feats = self.__load_features(fnames, self.limit_kpm)
        transformed_feats = self.transform_features(feats)
        self.save_features(transformed_feats, fnames)

    def run(self):

        # all_feats, all_fnames = self.load_all_features()
        # all_mid_level_feats = self.transform_features(all_feats)
        # self.save_features(all_mid_level_feats, all_fnames)
        # return all_mid_level_feats, all_fnames

        batch_size = 10240

        all_fnames = self._meta['all_fnames']
        n_batches = len(all_fnames)/batch_size

        if n_batches < 1:
            n_batches = 1

        split_fnames = np.array_split(all_fnames, n_batches, axis=0)

        for i, fnames in enumerate(split_fnames):
            feats = self.__load_features(fnames, self.limit_kpm)
            transformed_feats = self.transform_features(feats)
            self.save_features(transformed_feats, fnames)
            progressbar("", i+1, len(split_fnames))
