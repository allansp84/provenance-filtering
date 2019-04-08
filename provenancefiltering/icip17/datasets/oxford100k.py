# -*- coding: utf-8 -*-

import os
import csv
import itertools
import numpy as np

from glob import glob

from provenancefiltering.icip17.datasets.dataset import Dataset


class Oxford100k(Dataset):

    def __init__(self, dataset_path, output_path='./working', file_types=('jpg', 'png'),
                 groundtruth_path='', n_round=1, n_batches=4, query_id='', n_samples=0):

        super(Oxford100k, self).__init__(dataset_path, output_path, file_types)

        self.__meta_info = {}
        self.groundtruth_path = groundtruth_path
        self.n_round = n_round
        self.n_batches = n_batches
        self.query_id = query_id
        self.n_samples = n_samples

    @staticmethod
    def read_ground_truth(fname, delimiter=',', remove_header=True):

        # -- read the file content
        with open(fname, 'r') as f:
            csv_data = list(csv.reader(f, delimiter=delimiter))

        # -- removing header
        first_line = csv_data[0][0].strip()
        if '#' in first_line[0]:
            if remove_header:
                csv_data = csv_data[1:]

        # -- get just the related images with a given query
        csv_data = np.array(csv_data)
        idxs = np.where(csv_data[:, 3] == '1')[0]
        csv_data = csv_data[idxs]

        return csv_data

    def _build_meta(self, in_path, file_types):
        img_idx = 0

        all_fnames = []
        all_idxs = []
        all_img_id = []

        query_idxs = []
        search_idxs = []
        sub_class = []
        distractor_idxs = []

        filtered_csv = np.empty((0, 4), dtype=np.str)

        gt_all_idxs = {}
        hash_idx_probe = {}
        hash_idx_world = {}

        if self.groundtruth_path:
            filtered_csv = self.read_ground_truth(self.groundtruth_path, delimiter=' ', remove_header=False)

        if self.query_id:
            ss = np.where(filtered_csv == self.query_id)
            filtered_csv = filtered_csv[ss[0]]

        folders = [self._list_dirs(in_path, file_types)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))

        fnames = []
        for i, folder in enumerate(folders):
            ffns = [glob(os.path.join(in_path, folder, '*' + file_type)) for file_type in file_types]
            fnames += [list(itertools.chain.from_iterable(ffns))]
        fnames = np.concatenate(fnames)
        fnames = np.sort(fnames)

        probes_idxs = np.flatnonzero(np.char.find(fnames, 'queries') != -1)
        gallery_idxs = np.flatnonzero(np.char.find(fnames, 'queries') == -1)

        probes_fnames = fnames[probes_idxs]
        gallery_fnames = fnames[gallery_idxs]

        for fname in probes_fnames:

            rel_path = os.path.relpath(fname, in_path)
            img_id, ext = os.path.splitext(os.path.basename(rel_path))

            if ext in [".npy", ".kp"]:
                img_id = os.path.splitext(os.path.basename(rel_path))[0]
            else:
                img_id = os.path.basename(rel_path)

            probe_id = os.path.splitext(img_id)[0].split("_")[0]

            if probe_id in filtered_csv[:, 0]:
                try:
                    gt_all_idxs[probe_id] += [img_idx]
                except KeyError:
                    gt_all_idxs[probe_id] = [img_idx]

                hash_idx_probe.update({img_id: img_idx})
                all_img_id += [probe_id]
                all_fnames += [fname]
                all_idxs += [img_idx]
                query_idxs += [img_idx]
                sub_class += ['probe']
                img_idx += 1

        for fname in gallery_fnames:

            rel_path = os.path.relpath(fname, in_path)
            img_id, ext = os.path.splitext(os.path.basename(rel_path))

            if ext in [".npy", ".kp"]:
                img_id = os.path.splitext(os.path.basename(rel_path))[0]
            else:
                img_id = os.path.basename(rel_path)

            if self.groundtruth_path:
                ss = np.where(os.path.splitext(img_id)[0] == filtered_csv)
                for key in filtered_csv[ss[0], 0]:
                    gt_all_idxs[key] += [img_idx]

            hash_idx_world.update({img_id: img_idx})

            all_fnames += [fname]
            all_idxs += [img_idx]
            all_img_id += [os.path.splitext(img_id)[0]]
            search_idxs += [img_idx]
            sub_class += ['world']
            img_idx += 1

        all_fnames = np.array(all_fnames)
        all_idxs = np.array(all_idxs)
        query_idxs = np.array(query_idxs)
        distractor_idxs = np.array(distractor_idxs)
        sub_class = np.array(sub_class)

        n_data = len(all_idxs)
        scenes_id = np.zeros((len(gt_all_idxs), n_data), dtype=np.uint32)
        scenes_id_host = []
        scenes_id_alien = []


        if self.groundtruth_path:
            for i in range(len(gt_all_idxs)):
                y = np.zeros(all_idxs.shape, dtype=np.uint32)
                nd = gt_all_idxs[all_img_id[i]]
                y[nd] = 1
                scenes_id[i] = y

        search_idxs = np.setdiff1d(all_idxs, query_idxs)
        search_idxs = np.sort(search_idxs)

        r_state = np.random.RandomState(7)
        for n in range(self.n_round):
            search_idxs = r_state.permutation(search_idxs)
        splited_search_idxs = np.array_split(search_idxs, self.n_batches, axis=0)

        self.n_queries = len(query_idxs)
        self.n_gallery = len(search_idxs)
        print("-- all_fnames", all_fnames.shape, flush=True)
        print("-- query_idxs", query_idxs.shape, flush=True)
        print("-- search_idxs", search_idxs.shape, flush=True)

        print("-- related images per query (mean)", scenes_id.sum(axis=1).mean())

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': scenes_id,
                  's_all_labels': [(scenes_id, "All"), (scenes_id_host, "Host"), (scenes_id_alien, "Alien")],
                  'all_idxs': all_idxs,
                  'distractor_idxs': distractor_idxs,
                  'search_idxs': splited_search_idxs,
                  'all_search_idxs': np.sort(search_idxs),
                  'query_idxs': np.sort(query_idxs),
                  'sub_class': sub_class,
                  'hash_idx_probe': hash_idx_probe,
                  'hash_idx_world': hash_idx_world,
                  }

        return r_dict

    def meta_info_feats(self, output_path, file_types):
        return self._build_meta(output_path, file_types)

    def meta_info_images(self, output_path, file_types):
        return self._build_meta(output_path, file_types)
