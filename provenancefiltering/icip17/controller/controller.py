# -*- coding: utf-8 -*-

import os
import sys
import cv2
import time
import shutil
import numpy as np

from glob import glob
from operator import itemgetter

from provenancefiltering.icip17.controller import BaseController

from provenancefiltering.icip17.feature import FeatureExtraction
from provenancefiltering.icip17.feature import local_feature_detection_and_description
from provenancefiltering.icip17.feature import keypoints_from_array, keypoints_to_array
from provenancefiltering.icip17.midlevelfeatures import MidLevelFeatures

from provenancefiltering.icip17.indexing import create_rank_files, create_feature_index, search_index
from provenancefiltering.icip17.indexing import match

from provenancefiltering.icip17.subspaces import Subspaces
from provenancefiltering.icip17.subspaces import subspace_algo

from provenancefiltering.icip17.evaluation import Evaluation

from provenancefiltering.icip17.utils import *

from sklearn import metrics


class LoadFeatures(object):
    def __init__(self, input_fname, n_feats):
        self.input_fname = input_fname
        self.n_feats = n_feats

    def run(self):
        features = np.load(self.input_fname)[:self.n_feats, :]
        return [features.shape[0], features]


class Controller(BaseController):
    def __init__(self, data, args):

        super(Controller, self).__init__(data, args)

        self.data = data
        self.args = args
        self.features_path = "features"
        self.indexing_path = "indexing"
        self.matches_path = "matches"
        self.dirname_masks = "n_masks"

        self.subspace_path = ''

        if self.args.subspace_algo:
            self.subspace_path = "{0}{1}".format(subspace_algo[self.args.subspace_algo], self.args.n_components)

        self.features_query_contraction_path = "features_query_contraction"
        self.features_query_expansion_path = "features_query_expansion"

        if self.args.query_contraction:
            self.features_path = self.features_query_contraction_path
            self.indexing_path = "indexing_query_contraction"
            self.matches_path = "matches_query_contraction"
            self.args.subspace_algo = 0

        if self.args.query_expansion:
            self.features_path = self.features_query_contraction_path
            self.indexing_path = "indexing_query_expansion"
            self.matches_path = "matches_query_expansion"
            self.args.subspace_algo = 0

        self.descriptor = "Match{0}".format(self.args.descriptor_kpm)
        self.n_round = "round_{0}".format(self.args.n_round)

        mlf_search_space = {'CS': 'kmeans', 'SDD': 'unified', 'DS': 160, 'CP': 'softmax'}
        self.mlf_trials = [mlf_search_space]

    def load_features(self, fnames):

        print('-- loading features', flush=True)

        n_key_points_by_img = []
        n_fnames = len(fnames)
        all_features = []

        for i in range(n_fnames):
            feats = np.load(fnames[i])[:self.args.limit_kpm, :]
            all_features += [feats]
            n_key_points_by_img += [feats.shape[0]]
            progressbar('', i + 1, n_fnames)

        return np.concatenate(all_features), n_key_points_by_img

    def load_features_par(self, fnames):

        print('-- loading features', flush=True)

        tasks = []
        for i in range(len(fnames)):
            tasks += [LoadFeatures(fnames[i], self.args.limit_kpm)]

        print("-- running %d tasks in parallel (%d Jobs)" % (len(tasks), self.args.n_jobs), flush=True)
        output = RunInParallelWithReturn(tasks, self.args.n_jobs).run()

        n_key_points_by_img = [output[i][0] for i in range(len(output))]
        all_features = [output[i][1] for i in range(len(output))]

        return np.concatenate(all_features), n_key_points_by_img

    def load_features_metadata(self, input_path='', file_type='npy'):

        if not input_path:
            if self.args.subspace_algo:  # and (not self.args.query_contraction) and (not self.args.query_expansion):
                input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), "features_subspaces",
                                          self.subspace_path)
            else:
                input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.features_path)

        if 'kp' in file_type and self.args.subspace_algo:
            input_path = input_path.replace("features_subspaces", "features")
            input_path = os.path.dirname(input_path)

        meta_info_feats = self.data.meta_info_feats(input_path, [file_type])

        return meta_info_feats

    def gather_feature_vector_from_query(self, force_written=False):

        print("-- creating db features for queries", flush=True)
        sys.stdout.flush()

        start = get_time()

        if not self.args.subspace_algo and not self.args.query_contraction and not self.args.query_expansion:
            input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.features_path,
                                      )
        else:
            input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), "features_subspaces",
                                      self.subspace_path)

        output_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.indexing_path,
                                   self.n_round, self.subspace_path)

        safe_create_dir(output_path)

        prefix_kpm_query = "{0}_query".format(self.args.prefix_kpm)
        all_feats_output_path = "{0:s}/{1:s}.npy".format(output_path, prefix_kpm_query)
        indexing_file = "{0:s}/{1:s}_indexing.txt".format(output_path, prefix_kpm_query)

        if force_written or (not os.path.isfile(all_feats_output_path) and not os.path.isfile(indexing_file)):

            meta_info_feats = self.load_features_metadata(input_path=input_path)

            all_fnames = meta_info_feats['all_fnames']
            query_idxs = meta_info_feats['query_idxs']

            query_features, n_keypoints_by_img = self.load_features(all_fnames[query_idxs])

            count = 0
            feat_indexing = []

            for i, fname in enumerate(all_fnames[query_idxs]):
                imgname = os.path.basename(fname)
                count += n_keypoints_by_img[i]
                if WITH_INDEX:
                    feat_indexing.append(["{0}_{1}{2}".format(os.path.splitext(imgname)[0], query_idxs[i], os.path.splitext(imgname)[1]),
                                          n_keypoints_by_img[i], count])
                else:
                    feat_indexing.append(["{0}".format(imgname), n_keypoints_by_img[i], count])

            np.save(all_feats_output_path, query_features)
            self.write_feat_indexing(indexing_file, feat_indexing)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def gather_feature_vector_from_gallery(self):

        print("-- creating db features for the gallery", flush=True)
        sys.stdout.flush()

        start = get_time()

        if not self.args.subspace_algo and not self.args.query_contraction and not self.args.query_expansion:
            input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.features_path,
                                      )
        else:
            input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), "features_subspaces",
                                      self.subspace_path)

        meta_info_feats = self.load_features_metadata(input_path=input_path)

        all_fnames = meta_info_feats['all_fnames']
        split_search_idxs = meta_info_feats['search_idxs']

        output_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.indexing_path,
                                   self.n_round, self.subspace_path)

        safe_create_dir(output_path)

        for s, search_idxs in enumerate(split_search_idxs):

            prefix_kpm_search = "{0}_search_{1}".format(self.args.prefix_kpm, s)
            indexing_file = "{0:s}/{1:s}_indexing.txt".format(output_path, prefix_kpm_search)

            # if force_written or not os.path.exists(indexing_file):
            if not os.path.exists(indexing_file):

                count = 0
                feat_indexing = []
                gallery_fnames = all_fnames[search_idxs]
                batches_indexes = [bs for bs in range(0, len(gallery_fnames), DB_FEATS_BATCH_SIZE)]
                batches_indexes += [len(gallery_fnames)]
                for bs in range(len(batches_indexes) - 1):

                    all_feats_output_path = "{0:s}/{1:s}/batch_{2:03d}.npy".format(output_path, prefix_kpm_search, bs)

                    if not os.path.isfile(all_feats_output_path):

                        gallery_fnames_bs = gallery_fnames[batches_indexes[bs]:batches_indexes[bs + 1]]
                        search_features, n_keypoints_by_img = self.load_features_par(gallery_fnames_bs)

                        for i, fname in enumerate(gallery_fnames_bs):
                            imgname = os.path.basename(fname)
                            count += n_keypoints_by_img[i]
                            feat_indexing.append(["{0}".format(imgname), n_keypoints_by_img[i], count])

                        safe_create_dir(os.path.dirname(all_feats_output_path))

                        if 'npz' in self.args.db_filetype:
                            filename = all_feats_output_path.replace('.npy', '.npz')
                            np.savez_compressed(filename, search_features)
                        else:
                            np.save(all_feats_output_path, search_features)

                    else:
                        print(" -- {0} already exists".format(all_feats_output_path), flush=True)
                        sys.stdout.flush()

                if feat_indexing:
                    self.write_feat_indexing(indexing_file, feat_indexing)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def gather_feature_vectors(self):

        print("-- gather feature vectors ...", flush=True)

        # self.gather_feature_vector_from_query(force_written=self.args.force_written)
        self.gather_feature_vector_from_gallery()

    def feature_extraction(self):

        print("Feature extraction ...", flush=True)

        start = get_time()

        # -- extracting features
        input_fnames = self.data.meta_info['all_fnames']
        all_idxs = self.data.meta_info['all_idxs']
        query_idxs = self.data.meta_info['query_idxs']

        output_dirname = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.features_path)
        output_fnames = []
        for i in range(len(input_fnames)):
            rel_fname = os.path.relpath(input_fnames[i], self.data.dataset_path)
            rel_fname = '{0}.npy'.format(rel_fname)
            output_fnames.append(os.path.join(output_dirname, rel_fname))
        output_fnames = np.array(output_fnames)

        output_fnames_maps = []
        for i in range(len(input_fnames)):
            rel_fname = os.path.relpath(input_fnames[i], self.data.dataset_path)
            output_fnames_maps.append(os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), "maps", rel_fname))
        output_fnames_maps = np.array(output_fnames_maps)

        tasks = []
        for idx in range(len(input_fnames)):

            if self.args.use_map:
                tasks += [FeatureExtraction(input_fnames[idx], output_fnames[idx],
                                            self.args.detector_kpm, self.args.descriptor_kpm,
                                            self.args.limit_kpm,
                                            self.args.resize_img,
                                            force_written=False,
                                            use_map=bool(all_idxs[idx] in query_idxs),
                                            output_fnames_maps=output_fnames_maps[idx],
                                            default_params=self.args.default_params,
                                            )]

            else:
                tasks += [FeatureExtraction(input_fnames[idx], output_fnames[idx],
                                            self.args.detector_kpm, self.args.descriptor_kpm,
                                            self.args.limit_kpm,
                                            self.args.resize_img,
                                            force_written=False,
                                            use_map=False,
                                            output_fnames_maps=output_fnames_maps[idx],
                                            default_params=self.args.default_params,
                                            )]

        if self.args.n_jobs > 1:
            print("-- running %d tasks in parallel (%d Jobs)" % (len(tasks), self.args.n_jobs), flush=True)
            RunInParallel(tasks, self.args.n_jobs).run()
        else:
            print("-- running %d tasks in sequence" % len(tasks), flush=True)
            for idx in range(len(tasks)):
                t_start = time.time()
                tasks[idx].run()
                t_end = time.time()
                progressbar('-- RunInSequence', idx + 1, len(tasks))
                print("spent time: {0}!".format(t_end - t_start), flush=True)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def extract_mid_level_features(self):
        """ docstring """

        start = get_time()

        for trial in self.mlf_trials:

            input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm),
                                      self.features_path)

            metainfo_feats = self.data.meta_info_feats(input_path, ['npy'])
            output_path = os.path.join(input_path, trial["CS"], trial["SDD"], str(trial["DS"]))
            output_path = output_path.replace('low_level_features', 'mid_level_features')

            print("{0}/{1}".format(output_path, trial["CP"]))

            midlevelfeats = MidLevelFeatures(metainfo_feats, input_path, output_path,
                                             codebook_selection=trial["CS"],
                                             codebook_build=trial["SDD"],
                                             codebook_size=trial["DS"],
                                             coding_poling=trial["CP"],
                                             n_jobs=N_JOBS)

            midlevelfeats.build_codebook()
            midlevelfeats.run()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def extract_subspace_features(self):

        print("Building subspace ...", flush=True)

        start = get_time()

        input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.features_path)

        meta_info_feats = self.data.meta_info_feats(input_path, ['npy'])

        output_path = os.path.join(input_path, "{0}{1}".format(subspace_algo[self.args.subspace_algo], self.args.n_components))

        dirname_features = "/{0}/".format(self.features_path)
        output_path = output_path.replace(dirname_features, '/features_subspaces/')

        feature_subspace = Subspaces(meta_info_feats, input_path, output_path,
                                     algo=self.args.subspace_algo,
                                     descriptor=self.args.descriptor_kpm,
                                     n_components=self.args.n_components,
                                     limit_kpm=self.args.limit_kpm,
                                     n_jobs=self.args.n_jobs)

        feature_subspace.fit_subspace()
        feature_subspace.run()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def creating_indexes(self):

        print("Creating Indexes ...", flush=True)

        start = get_time()

        # -- creating indexes
        input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.indexing_path,
                                  self.n_round, self.subspace_path)

        # input_query_dbfeats = "{0:s}/{1:s}_query.npy".format(input_path, self.args.prefix_kpm)
        # output_query_index = os.path.join(input_path, "query")
        # create_feature_index(input_query_dbfeats, output_query_index, "index",
        #                      self.args.index_type_kpm, self.args.distance_kpm, self.args.lib_kpm, self.args.n_kdtree_kpm,
        #                      self.args.subsampling, self.args.niter_pq)

        split_search_idxs = self.data.meta_info['search_idxs']
        for s in range(len(split_search_idxs)):
            input_search_dbfeats = "{0:s}/{1:s}_search_{2}".format(input_path, self.args.prefix_kpm, s)
            input_search_dbfeats = retrieve_samples(input_search_dbfeats, self.args.db_filetype)

            output_search_index = os.path.join(input_path, "search_{0}".format(s))
            create_feature_index(input_search_dbfeats, output_search_index, "index",
                                 self.args.index_type_kpm, self.args.distance_kpm, self.args.lib_kpm, self.args.n_kdtree_kpm,
                                 self.args.subsampling, self.args.niter_pq, self.args.db_filetype)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def searching_indexes(self):

        print("Searching indexes ...", flush=True)

        start = get_time()

        # -- computing distances

        # input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.indexing_path,
        #                           self.n_round, self.subspace_path)

        # if self.args.subspace_algo and (not self.args.query_contraction) and (not self.args.query_expansion):
        #     input_path = os.path.join(input_path, self.subspace_path)

        meta_info_feats = self.load_features_metadata()

        all_fnames_feats = meta_info_feats['all_fnames']
        query_idxs = meta_info_feats['query_idxs']

        split_search_idxs = self.data.meta_info['search_idxs']
        for s in range(len(split_search_idxs)):

            distance_basename = "{0}_{1}_{2}_{3}".format(self.args.index_type_kpm,
                                                         self.args.distance_kpm,
                                                         self.args.n_neighbors_kpm,
                                                         self.args.rfactor_kpm,
                                                         )

            output_path = os.path.join(self.data.output_path,
                                       self.descriptor,
                                       str(self.args.limit_kpm),
                                       self.indexing_path,
                                       self.n_round,
                                       self.subspace_path,
                                       "dists",
                                       distance_basename,
                                       )

            if self.args.index_type_kpm in ["KDFOREST", "PQ", "LOPQ", "IVFPQ", "RIVFPQ"]:
                output_path = "{0}_{1}_{2}".format(output_path, self.args.n_kdtree_kpm, s)
            else:
                output_path = "{0}_{1}".format(output_path, s)

            input_search_fname = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.indexing_path,
                                              self.n_round, self.subspace_path)

            input_path_flann = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.indexing_path,
                                            self.n_round, self.subspace_path)

            input_search_dbfeats = "{0:s}/{1:s}_search_{2}".format(input_search_fname, self.args.prefix_kpm, s)

            if self.args.index_type_kpm in ["KDFOREST", "PQ", "LOPQ", "IVFPQ", "RIVFPQ"]:
                search_index_path = "{0:s}/search_{1}/{2:s}_{3:s}_{4:s}_{5}.dat".format(input_path_flann, s, "index",
                                                                                        self.args.index_type_kpm,
                                                                                        self.args.distance_kpm,
                                                                                        self.args.n_kdtree_kpm)
            else:
                search_index_path = "{0:s}/search_{1}/{2:s}_{3:s}_{4:s}.dat".format(input_path_flann, s, "index",
                                                                                    self.args.index_type_kpm,
                                                                                    self.args.distance_kpm)

            search_index_path = search_index_path.replace(self.indexing_path, "indexing")
            input_search_dbfeats = input_search_dbfeats.replace(self.indexing_path, "indexing")
            input_search_dbfeats = retrieve_samples(input_search_dbfeats, self.args.db_filetype)
            search_index(all_fnames_feats[query_idxs], input_search_dbfeats, search_index_path, output_path,
                         self.args.search_type_kpm, self.args.n_neighbors_kpm, self.args.rfactor_kpm, self.args.lib_kpm,
                         self.args.force_written, self.args.db_filetype)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def compute_mask_using_top_k(self, all_fnames, all_fnames_feats, all_fnames_kp, q_idx, top_k_idx):

        # -- load images
        query_img = cv2.imread(all_fnames[q_idx])[:, :, ::-1]
        top_one_img = cv2.imread(all_fnames[top_k_idx])[:, :, ::-1]

        # -- load SURF descriptors
        query_desc = np.load(all_fnames_feats[q_idx])
        top_one_desc = np.load(all_fnames_feats[top_k_idx])

        # -- load SURF key-points
        query_kp = np.loadtxt(all_fnames_kp[q_idx])
        top_one_kp = np.loadtxt(all_fnames_kp[top_k_idx])

        # -- filter the key-points to pick up just the good matches
        bf = cv2.BFMatcher()

        # -- sort them in the order of their distance.
        matches = bf.match(query_desc, top_one_desc)
        good_matches = sorted(matches, key=lambda x: x.distance)

        q_mkeyp, t_mkeyp, _ = match.get_matched_keypoints_from_list(good_matches[:25],
                                                                    keypoints_from_array(query_kp),
                                                                    keypoints_from_array(top_one_kp))

        if self.args.use_reference_mask:

            output_fname_cropped_img = os.path.join(UTILS_PATH, self.dirname_masks)

            if not os.path.exists(output_fname_cropped_img):
                print('Masks Not Found!', output_fname_cropped_img, flush=True)
                sys.exit(0)

                # query_img_id = os.path.splitext(os.path.basename(all_fnames_feats[q_idx]))[0]
                # output_fname_mask_img = "{0}/{1}.mask.png".format(output_fname_cropped_img, query_img_id)

                # filtered_map = cv2.imread(output_fname_mask_img, cv2.IMREAD_GRAYSCALE)

        else:

            # --  calculate homography
            h, status = cv2.findHomography(t_mkeyp, q_mkeyp, cv2.RANSAC)
            registered_img = cv2.warpPerspective(top_one_img, h, (query_img.shape[1], query_img.shape[0]))

            registered_img_gray = cv2.cvtColor(registered_img, cv2.COLOR_RGB2GRAY)
            query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)

            number_of_shades = 32
            factor_quantization = 255. / (number_of_shades - 1)

            registered_img_gray = np.round(registered_img_gray / factor_quantization)
            registered_img_gray *= factor_quantization
            registered_img_gray = np.array(registered_img_gray, dtype=np.uint8)

            query_img_gray = np.round(query_img_gray / factor_quantization)
            query_img_gray *= factor_quantization
            query_img_gray = np.array(query_img_gray, dtype=np.uint8)

            result_img = registered_img_gray - query_img_gray

            _, binary_img = cv2.threshold(result_img.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

            kernel = np.ones((11, 11), np.uint8)
            erosion = cv2.erode(binary_img, kernel, iterations=1)
            filtered_map = cv2.dilate(erosion, kernel, iterations=12)

            # filtered_map = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
            # filtered_map = cv2.morphologyEx(filtered_map, cv2.MORPH_CLOSE, kernel, iterations=8)

            filtered_map = cv2.medianBlur(filtered_map, 5)

            # plt.imshow(registered_img_gray, cmap='gray'); plt.title("Result (after erosion and dilation)"); plt.show()
            # plt.imshow(result_img, cmap='gray'); plt.title("Result (after erosion and dilation)"); plt.show()
            # plt.imshow(binary_img, cmap='gray'); plt.title("Result (after erosion and dilation)"); plt.show()
            # plt.imshow(filtered_map, cmap='gray'); plt.title("Result (after erosion and dilation)"); plt.show()

            return query_img, top_one_img, registered_img, filtered_map

    def context_based_rerank(self, distances_path):

        print("-- context-based query contraction ...", flush=True)
        sys.stdout.flush()

        dists = np.load(distances_path)

        query_idxs = self.data.meta_info['query_idxs']
        search_idxs = self.data.meta_info['all_search_idxs']
        all_fnames = self.data.meta_info['all_fnames']

        meta_info_feats = self.load_features_metadata(file_type='npy')
        all_fnames_feats = meta_info_feats['all_fnames']

        meta_info_feats = self.load_features_metadata(file_type='kp')
        all_fnames_kp = meta_info_feats['all_fnames']

        for i, q_idx in enumerate(query_idxs):

            top_one_idx = int(dists[q_idx, 0, 1])
            query_img, top_one_img, registered_img, filtered_map = self.compute_mask_using_top_k(all_fnames,
                                                                                                 all_fnames_feats,
                                                                                                 all_fnames_kp,
                                                                                                 q_idx,
                                                                                                 top_one_idx)

            key_points, feature_vectors, det_t, dsc_t = local_feature_detection_and_description("", self.args.detector_kpm,
                                                                                                self.args.descriptor_kpm,
                                                                                                kmax=self.args.limit_kpm,
                                                                                                img=query_img,
                                                                                                mask=filtered_map,
                                                                                                default_params=True,
                                                                                                )

            # im1keyp = cv2.drawKeypoints(query_img, key_points, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            # plt.imshow(im1keyp, cmap='gray'); plt.title("Result (after erosion and dilation)"); plt.show()

            if len(feature_vectors) == 0:
                feature_vectors = np.zeros((1, DESCRIPTOR_SIZE[self.args.descriptor_kpm]), dtype=DESCRIPTOR_TYPE[self.args.descriptor_kpm])
                key_points = np.zeros((1, 7), dtype=DESCRIPTOR_TYPE[self.args.descriptor_kpm])

            if isinstance(key_points[0], cv2.KeyPoint().__class__):
                key_points = keypoints_to_array(key_points)
            else:
                key_points = np.array(key_points, dtype=np.float32)

            if self.args.subspace_algo:
                input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.features_path)

                meta_info_feats = self.data.meta_info_feats(input_path, ['npy'])

                output_path = os.path.join(input_path, self.subspace_path)
                dirname_features = "/{0}/".format(self.features_path)
                output_path = output_path.replace(dirname_features, '/features_subspaces/')

                feature_subspace = Subspaces(meta_info_feats, input_path, output_path,
                                             algo=self.args.subspace_algo,
                                             descriptor=self.args.descriptor_kpm,
                                             n_components=self.args.n_components,
                                             limit_kpm=self.args.limit_kpm,
                                             n_jobs=self.args.n_jobs)

                feature_vectors = feature_subspace.transform_features([feature_vectors])
                feature_vectors = np.reshape(feature_vectors, (-1, self.args.n_components))

            if self.args.subspace_algo:
                features_path = 'features_subspaces'
            else:
                features_path = self.features_path

            dirname_features = "/{0}/".format(features_path)

            features_query_contraction_path = "/{0}/".format(self.features_query_contraction_path)
            output_fname = all_fnames_feats[q_idx].replace(dirname_features, features_query_contraction_path)

            key_point_path = "{0}.kp".format(os.path.splitext(output_fname)[0])

            safe_create_dir(os.path.dirname(output_fname))

            np.savetxt(key_point_path, key_points, fmt='%.4f')
            np.save(output_fname, feature_vectors)

            output_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.dirname_masks)
            safe_create_dir(output_path)

            query_img_id = os.path.splitext(os.path.basename(all_fnames_feats[q_idx]))[0]
            output_fname_mask_img = "{0}/{1}.mask.png".format(output_path, query_img_id)
            output_fname_query_img = "{0}/{1}.png".format(output_path, query_img_id)
            output_fname_top_one_img = "{0}/{1}.top_one.png".format(output_path, query_img_id)

            cv2.imwrite(output_fname_mask_img, filtered_map)
            cv2.imwrite(output_fname_query_img, query_img[:, :, ::-1])
            cv2.imwrite(output_fname_top_one_img, top_one_img[:, :, ::-1])

            print(output_fname, feature_vectors.shape, flush=True)
            sys.stdout.flush()

            progressbar('-- query', i + 1, len(query_idxs))

        # -- copy world set
        for i, g_idx in enumerate(search_idxs):

            if self.args.subspace_algo:
                features_path = 'features_subspaces'
            else:
                features_path = self.features_path

            dirname_features = "/{0}/".format(features_path)

            features_query_contraction_path = "/{0}/".format(self.features_query_contraction_path)
            output_fname = all_fnames_feats[g_idx].replace(dirname_features, features_query_contraction_path)

            safe_create_dir(os.path.dirname(output_fname))

            if self.args.subspace_algo:
                features_path = 'features_subspaces'
                features_path = os.path.join(features_path, self.subspace_path)

            input_key_point_path = "{0}.kp".format(os.path.splitext(all_fnames_feats[g_idx])[0])
            output_key_point_path = "{0}.kp".format(os.path.splitext(output_fname)[0])
            shutil.copyfile(input_key_point_path.replace(features_path, 'features'), output_key_point_path)

            shutil.copyfile(all_fnames_feats[g_idx], output_fname)
            progressbar('-- gallery', i + 1, len(search_idxs))

            print(output_fname, flush=True)
            sys.stdout.flush()

    def filtering_matches(self, distances_path):

        print("-- query contraction", flush=True)
        sys.stdout.flush()

        dists = np.load(distances_path)

        query_idxs = self.data.meta_info['query_idxs']
        search_idxs = self.data.meta_info['all_search_idxs']

        meta_info_feats = self.load_features_metadata(file_type='npy')
        all_fnames_feats = meta_info_feats['all_fnames']

        meta_info_feats = self.load_features_metadata(file_type='kp')
        all_fnames_kp = meta_info_feats['all_fnames']

        for q_idx in query_idxs:

            top_one_idx = int(dists[q_idx, 0, 1])

            query_desc = np.load(all_fnames_feats[q_idx])
            top_one_desc = np.load(all_fnames_feats[top_one_idx])

            query_kp = np.loadtxt(all_fnames_kp[q_idx])

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(query_desc, top_one_desc, k=2)

            good_matches, good_matches_idxs = [], []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])
                    good_matches_idxs.append(m.queryIdx)

            dirname_features = "/{0}/".format(self.features_path)

            if 'bad' in self.args.filter_matches:

                all_idxs = range(len(query_kp))
                filtered_idxs = np.setdiff1d(all_idxs, good_matches_idxs)
                features_query_contraction_path = "/{0}/".format(self.features_query_contraction_path)
                output_fname = all_fnames_feats[q_idx].replace(dirname_features, features_query_contraction_path)

            else:

                filtered_idxs = good_matches_idxs
                features_query_expansion_path = "/{0}/".format(self.features_query_expansion_path)
                output_fname = all_fnames_feats[q_idx].replace(dirname_features, features_query_expansion_path)

            key_point_path = "{0}.kp".format(os.path.splitext(output_fname)[0])

            safe_create_dir(os.path.dirname(output_fname))

            np.savetxt(key_point_path, query_kp[filtered_idxs], fmt='%.4f')
            np.save(output_fname, query_desc[filtered_idxs])

        # -- copy world set
        for g_idx in search_idxs:

            dirname_features = "/{0}/".format(self.features_path)

            if 'bad' in self.args.filter_matches:
                features_query_contraction_path = "/{0}/".format(self.features_query_contraction_path)
                output_fname = all_fnames_feats[g_idx].replace(dirname_features, features_query_contraction_path)

            else:
                features_query_expansion_path = "/{0}/".format(self.features_query_expansion_path)
                output_fname = all_fnames_feats[g_idx].replace(dirname_features, features_query_expansion_path)

            safe_create_dir(os.path.dirname(output_fname))

            input_key_point_path = "{0}.kp".format(os.path.splitext(all_fnames_feats[g_idx])[0])
            output_key_point_path = "{0}.kp".format(os.path.splitext(output_fname)[0])
            shutil.copyfile(input_key_point_path, output_key_point_path)

            shutil.copyfile(all_fnames_feats[g_idx], output_fname)

    def creating_ranks(self):

        print("Creating ranks ...", flush=True)

        start = get_time()

        # -- creating ranks
        input_path = os.path.join(self.data.output_path, self.descriptor, str(self.args.limit_kpm), self.indexing_path,
                                  self.n_round, self.subspace_path)

        # if self.args.subspace_algo:
        #     input_path = os.path.join(input_path, self.subspace_path)

        split_search_idxs = self.data.meta_info['search_idxs']
        for s in range(len(split_search_idxs)):

            search_indexing_path = "{0:s}/{1:s}_search_{2}_indexing.txt".format(input_path, self.args.prefix_kpm, s)

            distance_basename = "{0}_{1}_{2}_{3}".format(self.args.index_type_kpm,
                                                         self.args.distance_kpm,
                                                         self.args.n_neighbors_kpm,
                                                         self.args.rfactor_kpm,
                                                         )

            dists_input_path = os.path.join(self.data.output_path,
                                            self.descriptor,
                                            str(self.args.limit_kpm),
                                            self.indexing_path,
                                            self.n_round,
                                            self.subspace_path,
                                            "dists",
                                            distance_basename,
                                            )

            if self.args.index_type_kpm in ["KDFOREST", "PQ", "LOPQ", "IVFPQ", "RIVFPQ"]:
                dists_input_path = "{0}_{1}_{2}".format(dists_input_path, self.args.n_kdtree_kpm, s)
            else:
                dists_input_path = "{0}_{1}".format(dists_input_path, s)

            if self.args.query_contraction or self.args.query_expansion:
                search_indexing_path = search_indexing_path.replace(self.indexing_path, "indexing")

            create_rank_files(dists_input_path, search_indexing_path, self.args.score_type_kpm,
                              limit=2000, force_written=self.args.force_written)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def merging_file_ranks_v1(self, franks_a, franks_b, mtype='sum', output='', n_gallery=2000):

        print("-- merging the files ranks", flush=True)
        sys.stdout.flush()

        start = get_time()

        final_franks = []

        for frank_a, frank_b in zip(franks_a, franks_b):

            start = get_time()

            scores_a = np.genfromtxt(frank_a, dtype=np.str, delimiter=',', max_rows=n_gallery)
            scores_b = np.genfromtxt(frank_b, dtype=np.str, delimiter=',', max_rows=n_gallery)

            query_img_id = "_".join(os.path.splitext(os.path.basename(frank_a))[0].split("_")[1:])

            print(query_img_id, flush=True)
            sys.stdout.flush()

            img_idxs = [np.where(scores_a[:, 0] == i)[0] for i in scores_b[:, 0]]

            frerank = []
            for pos_b, line in enumerate(scores_b):

                img_id = img_idxs[pos_b]

                score_b = float(line[1])
                norm_score_b = float(line[2])

                re_rank, norm_re_rank = score_b, norm_score_b

                if 'pos' in mtype:
                    re_rank, norm_re_rank = pos_b + 1, pos_b + 1

                if len(img_id):

                    score_a = float(scores_a[img_id, 1][0])
                    norm_score_a = float(scores_a[img_id, 2][0])

                    if 'sum' in mtype:
                        re_rank = score_a + score_b
                        norm_re_rank = norm_score_a + norm_score_b

                    elif 'max' in mtype:
                        re_rank = max(score_a, score_b)
                        norm_re_rank = max(norm_score_a, norm_score_b)

                    elif 'min' in mtype:
                        re_rank = min(score_a, score_b)
                        norm_re_rank = min(norm_score_a, norm_score_b)

                    elif 'mean' in mtype:
                        re_rank = (score_a + score_b) / 2.
                        norm_re_rank = (norm_score_a + norm_score_b) / 2.

                    elif 'prod' in mtype:
                        re_rank = score_a * score_b
                        norm_re_rank = norm_score_a * norm_score_b

                    elif 'pos' in mtype:
                        re_rank = (pos_b + 1) + (img_id[0] + 1)  # pow(pow((pos_b + 1), 2.0) + pow((img_id[0] + 1), 2.0), 1/2.)

                    else:
                        print("experimental", flush=True)
                        re_rank = score_a + score_b
                        norm_re_rank = abs(norm_score_a - norm_score_b)
                else:
                    pass

                frerank += [[line[0], re_rank, norm_re_rank]]
                print("", flush=True)

            if 'norm' in mtype:
                frerank = sorted(frerank, key=itemgetter(2), reverse=True)
            else:
                if 'pos' in mtype:
                    frerank = sorted(frerank, key=itemgetter(1), reverse=False)
                else:
                    frerank = sorted(frerank, key=itemgetter(1), reverse=True)

            output_fname = os.path.join(output, self.args.score_type_kpm, os.path.basename(frank_a))

            safe_create_dir(os.path.dirname(output_fname))

            np.savetxt(output_fname, frerank, fmt='%s', delimiter=',')
            final_franks += [output_fname]

            print("Input 1 {0}".format(frank_a), flush=True)
            print("Input 2 {0}".format(frank_b), flush=True)
            print("Output {0}".format(output_fname), flush=True)
            sys.stdout.flush()

            elapsed = total_time_elapsed(start, get_time())
            print(elapsed, flush=True)
            print("", flush=True)
            sys.stdout.flush()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

        return final_franks

    def merging_file_ranks_v2(self, franks_a, franks_b, mtype='sum', output='', n_gallery=2000):

        print("-- merging the files ranks", flush=True)
        sys.stdout.flush()

        start = get_time()

        hash_idx = self.data.meta_info['hash_idx']
        all_labels = self.data.meta_info['all_labels']

        final_franks = []
        kappa_coef, q_stat, jaccard_coef, skalak_coef, mutual_info, kendall, measures = [], [], [], [], [], [], []

        for frank_a, frank_b in zip(franks_a, franks_b):

            start = get_time()

            scores_a = np.genfromtxt(frank_a, dtype=np.str, delimiter=',', max_rows=n_gallery)
            scores_b = np.genfromtxt(frank_b, dtype=np.str, delimiter=',', max_rows=n_gallery)

            # scores_a = np.loadtxt(frank_a, dtype=np.str, delimiter=',')
            # scores_b = np.loadtxt(frank_b, dtype=np.str, delimiter=',')

            query_img_id = "_".join(os.path.splitext(os.path.basename(frank_a))[0].split("_")[1:])
            query_idx = hash_idx[query_img_id]

            print(query_img_id, flush=True)
            sys.stdout.flush()

            indexes_a = [hash_idx[score[0]] for score in scores_a]
            indexes_b = [hash_idx[score[0]] for score in scores_b]

            labels_a = all_labels[query_idx][indexes_a]
            labels_b = all_labels[query_idx][indexes_b]

            # -- Cohen Kappa Score
            kappa_coef += [metrics.cohen_kappa_score(labels_a, labels_b)]

            # -- Q-statistic
            cm = metrics.confusion_matrix(labels_a, labels_b)
            if float(cm[0, 0] * cm[1, 1] + cm[0, 1] * cm[1, 0]) == 0.:
                q_stat += [0.0]
            else:
                q_stat += [(cm[0, 0] * cm[1, 1] - cm[0, 1] * cm[1, 0]) / float(cm[0, 0] * cm[1, 1] + cm[0, 1] * cm[1, 0])]

            # -- Jaccard Coefficient
            jaccard_coef += [metrics.jaccard_similarity_score(scores_a[:, 0], scores_b[:, 0])]

            n_elements = len(labels_a)
            agreements = np.equal(labels_a, labels_b)
            n_agreements = np.count_nonzero(agreements)
            n_disagreements = np.size(agreements) - np.count_nonzero(agreements)
            skalak_coef += [n_disagreements / float(n_agreements + n_disagreements)]

            kendall += [(n_agreements - n_disagreements) / (n_elements * (n_elements - 1) / 2.)]

            mutual_info += [metrics.adjusted_mutual_info_score(labels_a, labels_b)]

            # measures += [stats.pearsonr(scores_a[:,2].astype(np.float32), scores_b[:,2].astype(np.float32))[0]]
            measures += [metrics.adjusted_mutual_info_score(labels_a, labels_b)]

            # -- hack to get the normalized ranks
            scores = scores_b[:, 1].astype(np.float32)
            norm = (scores - scores.min()) / (scores.max() - scores.min())
            norm = np.reshape(norm, (-1, 1))
            scores_b = np.concatenate((scores_b, norm.astype(np.str)), axis=1)

            # img_a_idxs = [np.where(scores_b == i)[0] for i in scores_a[:, 0]]

            img_b_idxs = []
            for i in scores_b[:, 0]:
                img_b_idxs += [np.where(scores_a[:, 0] == i)[0]]

            frerank = []

            for pos_b, line in enumerate(scores_b):

                img_id = img_b_idxs[pos_b]

                score_b = float(line[1])
                norm_score_b = float(line[2])

                re_rank, norm_re_rank = score_b, norm_score_b

                if 'pos' in mtype:
                    re_rank, norm_re_rank = pos_b + 1, pos_b + 1

                if len(img_id):  # and (measures[-1] > 0.):

                    print(img_id, flush=True)

                    try:
                        score_a = float(scores_b[img_id, 1][0])
                        norm_score_a = float(scores_b[img_id, 2][0])
                    except IndexError:
                        score_a = float(scores_b[-1, 1])
                        norm_score_a = float(scores_b[-1, 2])

                    # print(complementarity_score,, flush=True)

                    if 'sum' in mtype:
                        re_rank = score_a + score_b
                        norm_re_rank = norm_score_a + norm_score_b

                    elif 'max' in mtype:
                        re_rank = max(score_a, score_b)
                        norm_re_rank = max(norm_score_a, norm_score_b)

                    elif 'min' in mtype:
                        re_rank = min(score_a, score_b)
                        norm_re_rank = min(norm_score_a, norm_score_b)

                    elif 'mean' in mtype:
                        re_rank = (score_a + score_b) / 2.
                        norm_re_rank = (norm_score_a + norm_score_b) / 2.

                    elif 'prod' in mtype:
                        re_rank = score_a * score_b
                        norm_re_rank = norm_score_a * norm_score_b

                    elif 'pos' in mtype:
                        re_rank = ((pos_b + 1) + (img_id[0] + 1)) / 2.
                        # re_rank = (score_b*(pos_b + 1) + score_a*(img_id[0] + 1))/2.
                        # re_rank = pow((pow(pos_b+1, 2.) + pow(img_id[0]+1, 2.)), 1/2.)
                        # pdb.set_trace()
                        # re_rank = distance.correlation([pos_b+1], [img_id[0]+1])
                        # re_rank = distance.euclidean([pos_b+1], [img_id[0]+1])

                    else:
                        pass
                else:
                    # print("Image not found {0}".format(line[0]), flush=True)
                    # sys.stdout.flush()
                    pass

                frerank += [[line[0], re_rank, norm_re_rank]]

            if 'norm' in mtype:
                frerank = sorted(frerank, key=itemgetter(2), reverse=True)
            else:
                if 'pos' in mtype:
                    frerank = sorted(frerank, key=itemgetter(1), reverse=False)
                else:
                    frerank = sorted(frerank, key=itemgetter(1), reverse=True)

            output_fname = os.path.join(output, self.args.score_type_kpm, os.path.basename(frank_a))

            safe_create_dir(os.path.dirname(output_fname))

            # output_fname = "{0}_{1}{2}".format(os.path.splitext(rank_fname)[0], mtype,
            #                                    os.path.splitext(rank_fname)[1])

            np.savetxt(output_fname, frerank, fmt='%s', delimiter=',')
            final_franks += [output_fname]

            print("Input A {0}".format(frank_a), flush=True)
            print("Input B {0}".format(frank_b), flush=True)
            print("Output {0}".format(output_fname), flush=True)
            sys.stdout.flush()

            elapsed = total_time_elapsed(start, get_time())
            print(elapsed, flush=True)
            print("", flush=True)
            sys.stdout.flush()

        np.savetxt("{0}/kappa_coef.txt".format(output), np.array(kappa_coef), fmt='%.5f')
        np.savetxt("{0}/q_stat.txt".format(output), np.array(q_stat), fmt='%.5f')
        np.savetxt("{0}/jaccard_coef.txt".format(output), np.array(jaccard_coef), fmt='%.5f')
        np.savetxt("{0}/skalak_coef.txt".format(output), np.array(skalak_coef), fmt='%.5f')
        np.savetxt("{0}/mutual_info.txt".format(output), np.array(mutual_info), fmt='%.5f')
        np.savetxt("{0}/kendall.txt".format(output), np.array(kendall), fmt='%.5f')
        np.savetxt("{0}/measures.txt".format(output), np.array(measures), fmt='%.5f')

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

        return final_franks

    def merge_indexing_methods(self, all_franks, mtype, output):
        print("-- merging indexing methods", flush=True)
        sys.stdout.flush()

        return self.merging_file_ranks_v2(all_franks[0, :], all_franks[1, :], mtype=mtype, output=output, n_gallery=2000)

    def saving_distances_matrix(self, n_top_k=2000):

        print("Saving Distance Matrix ...", flush=True)

        start = get_time()

        # -- prepare the input and output paths to saving distances
        dists_input_path = os.path.join(self.data.output_path, "{0}/{1}/{2}/{3}/{4}/dists/{5}_{6}_{7}_{8}".format(
            self.descriptor, self.args.limit_kpm, self.indexing_path, self.n_round,
            self.subspace_path, self.args.index_type_kpm, self.args.distance_kpm, self.args.n_neighbors_kpm,
            self.args.rfactor_kpm))

        if self.args.index_type_kpm in ["KDFOREST", "PQ", "LOPQ", "IVFPQ", "RIVFPQ"]:
            dists_input_path = "{0}_{1}".format(dists_input_path, self.args.n_kdtree_kpm)

        # -- reading metadata of the dataset
        query_idxs = self.data.meta_info['query_idxs']
        split_search_idxs = self.data.meta_info['search_idxs']
        hash_idx_probe = self.data.meta_info['hash_idx_probe']
        hash_idx_world = self.data.meta_info['hash_idx_world']

        r_dict = {idx: [] for idx in query_idxs}
        for s in range(len(split_search_idxs)):

            rank_files_path = "{0}_{1}/{2}".format(dists_input_path, s, self.args.score_type_kpm)
            franks = sorted(glob("{0}/*.rk".format(rank_files_path)))

            if self.args.merge_indexing_methods:
                all_rank_files_path = os.path.join(self.data.output_path, "{0}/{1}/{2}/{3}/{4}/dists".format(
                    self.descriptor, self.args.limit_kpm, self.indexing_path, self.n_round, self.subspace_path))

                all_franks = retrieve_samples(all_rank_files_path, 'rk')

                all_franks_kdforest = np.array([a for a in all_franks if 'KDFOREST' in a])
                all_franks_sfc = np.array([a for a in all_franks if 'SPACEFILLINGCURVES' in a])

                all_franks = np.concatenate((all_franks_kdforest.reshape((1, -1)), all_franks_sfc.reshape((1, -1))), axis=0)

                index_type_kpm = 'FusionIndexingMethods_{0}'.format(self.args.merge_type)
                dists_input_path = dists_input_path.replace(self.args.index_type_kpm, index_type_kpm)

                franks = self.merge_indexing_methods(all_franks, self.args.merge_type, dists_input_path)

            if self.args.merge_file_ranks:
                rank_files_path = rank_files_path.replace(self.indexing_path, "indexing")
                franks_b = sorted(glob("{0}/*.rk".format(rank_files_path)))

                index_type_kpm = '{0}_{1}'.format(self.args.index_type_kpm, self.args.merge_type)
                dists_input_path = dists_input_path.replace(self.args.index_type_kpm, index_type_kpm)

                franks = self.merging_file_ranks_v1(franks, franks_b, mtype=self.args.merge_type, output=dists_input_path)

            for i, frank in enumerate(franks):
                scores = np.genfromtxt(frank, dtype=np.str, delimiter=',', max_rows=n_top_k)

                dist = []
                for score in scores:
                    idx = hash_idx_world[score[0]]
                    dist += [[float(score[1]), idx]]

                basename = os.path.splitext(os.path.basename(frank))[0]
                key = "_".join(basename.split('_')[1:])
                key = hash_idx_probe[key]

                r_dict[key] += dist

                # print('saving the rank curve for query {0}'.format(i), flush=True)
                # sys.stdout.flush()
                # self.plot_rank_curve(scores, frank, top_k=100)

        dists = []
        for key in r_dict.keys():
            dists += [r_dict[key]]

        dists = np.array(dists, dtype=np.float32)

        # if self.args.n_batches != 1:
        #
        #     n_queries, n_row, n_cols = dists.shape[:3]
        #     n_batches = self.args.n_batches

        #     dists = np.reshape(np.reshape(dists, (n_queries, -1, (n_row / n_batches), n_cols)).swapaxes(1, 2), (n_queries, -1, n_cols))
        #     positions = 1. / (np.arange(n_row) + 1)
        #     positions = np.reshape(positions, (1, -1))
        #     dists[:, :, 0] = np.repeat(positions, n_queries, axis=0)

        output_path = "{0}_{1}.npy".format(dists_input_path, self.args.score_type_kpm)

        np.save(output_path, dists)

        print(output_path, flush=True)
        sys.stdout.flush()

        output_path = "{0}_{1}.npy".format(dists_input_path, self.args.score_type_kpm)

        if self.args.filter_matches in ['good', 'bad']:
            self.filtering_matches(output_path)

            print('-- finishing the filtering matches!', flush=True)
            sys.stdout.flush()
            sys.exit(1)

        if self.args.context_based_rerank:
            self.context_based_rerank(output_path)

            print('-- finishing the context-based re-ranking!', flush=True)
            sys.stdout.flush()
            sys.exit(1)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

    def computing_distances(self):

        print("Compute Distances ...", flush=True)

        if self.args.merge_indexing_methods:
            self.saving_distances_matrix()

        else:

            if not self.args.query_contraction and not self.args.query_expansion:
                self.gather_feature_vectors()
                self.creating_indexes()

            self.searching_indexes()

            self.creating_ranks()

            self.saving_distances_matrix()

    def classification(self):

        print("Matching ...", flush=True)

        start = get_time()

        meta_info_feats = self.load_features_metadata()

        descriptor = "Match{0}".format(self.args.descriptor_kpm)

        if self.args.index_type_kpm in ["KDFOREST", "PQ", "LOPQ", "IVFPQ", "RIVFPQ"]:
            distance_fname = "{0}_{1}_{2}_{3}_{4}_{5}.npy".format(self.args.index_type_kpm,
                                                                  self.args.distance_kpm,
                                                                  self.args.n_neighbors_kpm,
                                                                  self.args.rfactor_kpm,
                                                                  self.args.n_kdtree_kpm,
                                                                  self.args.score_type_kpm)

        else:
            distance_fname = "{0}_{1}_{2}_{3}_{4}.npy".format(self.args.index_type_kpm,
                                                              self.args.distance_kpm,
                                                              self.args.n_neighbors_kpm,
                                                              self.args.rfactor_kpm,
                                                              self.args.score_type_kpm)

        dists_input_path = os.path.join(self.data.output_path,
                                        descriptor,
                                        str(self.args.limit_kpm),
                                        self.indexing_path,
                                        self.n_round,
                                        self.subspace_path,
                                        "dists",
                                        distance_fname,
                                        )

        distance_basename = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.args.index_type_kpm,
                                                             self.args.distance_kpm,
                                                             self.args.n_neighbors_kpm,
                                                             self.args.rfactor_kpm,
                                                             self.args.n_kdtree_kpm,
                                                             self.args.score_type_kpm)

        output_path = os.path.join(self.data.output_path,
                                   descriptor,
                                   str(self.args.limit_kpm),
                                   self.matches_path,
                                   self.n_round,
                                   self.subspace_path,
                                   distance_basename,
                                   )

        if self.args.merge_file_ranks:
            index_type_kpm = '{0}_{1}'.format(self.args.index_type_kpm, self.args.merge_type)
            output_path = output_path.replace(self.args.index_type_kpm, index_type_kpm)
            dists_input_path = dists_input_path.replace(self.args.index_type_kpm, index_type_kpm)

        elif self.args.merge_indexing_methods:
            index_type_kpm = 'FusionIndexingMethods_{0}'.format(self.args.merge_type)
            output_path = output_path.replace(self.args.index_type_kpm, index_type_kpm)
            dists_input_path = dists_input_path.replace(self.args.index_type_kpm, index_type_kpm)

        else:
            pass

        lpredictions = Evaluation(output_path, meta_info_feats, descriptor,
                                  dists_path=dists_input_path,
                                  n_job=self.args.n_jobs,
                                  ).run()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()

        # _output_path = os.path.join(output_path, 'All')
        self.show_top_k_images_per_query(lpredictions['All'], output_path, top_k=25, n_query=20)
        # self.show_top_k_images_per_query(lpredictions['All'], output_path, top_k=100, n_query=10)
        # self.save_ranked_images_per_query(predictions, output_path)
        self.plot_precision_recall_at_ks(output_path)

    def execute_protocol(self):

        start = get_time()

        if self.args.n_batches != 1:
            # n_gallery = len(self.data.meta_info['all_search_idxs'])
            # divisors = np.array([x for x in range(1, n_gallery + 1) if not n_gallery % x])
            # new_n_batches = divisors[np.abs(divisors - self.args.n_batches).argmin()]
            #
            # if new_n_batches != self.args.n_batches:
            #     print("-- changing the number of batches to {0}".format(new_n_batches), flush=True)
            #     self.args.n_batches = new_n_batches
            pass

        self.data.n_batches = self.args.n_batches

        self.data.n_round = self.args.n_round

        self.data.query_id = self.args.query_id

        if self.args.feature_extraction:
            self.feature_extraction()
            self.extract_mid_level_features()

            if self.args.subspace_algo:
                self.extract_subspace_features()

        if self.args.compute_distances_kpm:
            self.computing_distances()

        if self.args.matching:
            self.classification()

        if self.args.plot_pr_curves:
            self.plot_precision_recall_curves(mean_curve=self.args.mean_curves)

        elapsed = total_time_elapsed(start, get_time())
        print('Total spent time: {0}!'.format(elapsed), flush=True)
        sys.stdout.flush()
