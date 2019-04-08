# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob

from provenancefiltering.icip17.utils import *
from provenancefiltering.icip17.datasets.dataset import Dataset


class Nimble2017(Dataset):

    def __init__(self, dataset_path, output_path='./working', file_types=('jpg', 'JPG', 'png', 'PNG', 'gif'),
                 n_round=1, n_batches=4, query_id='', n_samples=0):

        super(Nimble2017, self).__init__(dataset_path, output_path, file_types)

        self.__meta_info = {}
        self.n_round = n_round
        self.n_batches = n_batches
        self.query_id = query_id
        self.n_samples = n_samples

    def _build_meta(self, in_path, file_types, random_state=None, fold=None, check_files=False):

        img_idx = 0

        all_fnames = []
        all_idxs = []
        all_img_id = []

        query_idxs = []
        search_idxs = []
        sub_class = []
        distractor_idxs = []

        # fin = open(NC2017_Dev1_Beta4_gt, 'r')
        # csv_splice = np.array([line.split('\n')[0].split('|') for line in fin.readlines()])
        # fin.close()
        # filtered_csv = np.array([[line[0], line[1], line[2]] for line in csv_splice[1:]])

        NC2017_Dev1_Beta4 = "{0}/reference/provenance/NC2017_Dev1-provenance-ref-node.csv".format(self.dataset_path)
        fin = open(NC2017_Dev1_Beta4, 'r')
        csv_file = np.array([line.split('\n')[0].split('|') for line in fin.readlines()])
        fin.close()

        filtered_csv = []
        for line in csv_file[1:]:
            filtered_csv += [[line[0], line[1], line[1]]]
        filtered_csv = np.array(filtered_csv)
        # np.savetxt('NC2017_Dev1_Beta4-ground-truth.txt', filtered_csv, fmt='%s', delimiter='|', header='probe|alien|host')

        if self.query_id:
            ss = np.where(filtered_csv == self.query_id)
            filtered_csv = filtered_csv[ss[0]]

        hash_idx_probe = {}
        hash_idx_world = {}
        gt_all_idxs = {key: [] for key in filtered_csv[:, 0]}
        gt_alien_idxs = {key: [] for key in filtered_csv[:, 0]}
        gt_host_idxs = {key: [] for key in filtered_csv[:, 0]}
        donor_mask_fnames = {key: [] for key in filtered_csv[:, 0]}

        folders = [self._list_dirs(in_path, file_types)]
        folders = itertools.chain.from_iterable(folders)
        folders = sorted(list(folders))

        for i, folder in enumerate(folders):
            fnames = [glob(os.path.join(in_path, folder, '*' + file_type)) for file_type in file_types]
            fnames = itertools.chain.from_iterable(fnames)
            fnames = sorted(list(fnames))

            for fname in fnames:

                rel_path = os.path.relpath(fname, in_path)

                if WITH_FEAT_INDEX:
                    if os.path.splitext(fname)[1] in [".npy", ".bfv", ".fv", ".kp"]:
                        img_id = "_".join(os.path.splitext(os.path.basename(rel_path))[0].split("_")[:-1])
                    else:
                        img_id = os.path.splitext(os.path.basename(rel_path))[0]
                else:
                    img_id = os.path.splitext(os.path.basename(rel_path))[0]

                if 'probe' in rel_path:

                    if img_id in gt_all_idxs:

                        hash_idx_probe.update({img_id: img_idx})

                        gt_all_idxs[img_id] += [img_idx]
                        all_img_id += [img_id]

                        all_fnames += [fname]
                        all_idxs += [img_idx]
                        query_idxs += [img_idx]
                        sub_class += ['probe']
                        img_idx += 1

                else:

                    if 'world' in rel_path:

                        hash_idx_world.update({img_id: img_idx})

                        ss = np.where(filtered_csv == img_id)
                        for key in filtered_csv[ss[0], 0]:
                            gt_all_idxs[key] += [img_idx]

                        if len(ss[1]) > 0:
                            if ss[1][0] == 1:
                                for key in filtered_csv[ss[0], 0]:
                                    gt_alien_idxs[key] += [img_idx]

                            elif ss[1][0] == 2:
                                for key in filtered_csv[ss[0], 0]:
                                    gt_host_idxs[key] += [img_idx]

                        all_fnames += [fname]
                        all_idxs += [img_idx]
                        all_img_id += [img_id]
                        search_idxs += [img_idx]
                        sub_class += ['world']
                        img_idx += 1

                    elif 'reference' in rel_path:

                        ss = np.where(filtered_csv == img_id)
                        query_set = set(filtered_csv[ss[0], 0])
                        for key in query_set:
                            if "splice" in rel_path:
                                donor_mask_fnames[key] += [fname]

                    elif 'distractors' in rel_path:

                        if not self.n_samples or len(distractor_idxs) < self.n_samples:
                            hash_idx_world.update({img_id: img_idx})

                            all_fnames += [fname]
                            all_idxs += [img_idx]
                            distractor_idxs += [img_idx]
                            sub_class += ['world1M']
                            all_img_id += [img_id]
                            img_idx += 1

                    else:
                        pass

        all_fnames = np.array(all_fnames)
        all_idxs = np.array(all_idxs)
        query_idxs = np.array(query_idxs)
        search_idxs = np.array(search_idxs)
        distractor_idxs = np.array(distractor_idxs)
        sub_class = np.array(sub_class)

        n_data = len(all_idxs)

        scenes_id = np.zeros((len(gt_all_idxs), n_data), dtype=np.uint32)
        for i in range(len(gt_all_idxs)):
            y = np.zeros(all_idxs.shape, dtype=np.uint32)
            nd = gt_all_idxs[all_img_id[i]]
            y[nd] = 1
            scenes_id[i] = y

        scenes_id_alien = np.zeros((len(gt_alien_idxs), n_data), dtype=np.uint32)
        for i in range(len(gt_alien_idxs)):
            y = np.zeros(all_idxs.shape, dtype=np.uint32)
            nd = gt_alien_idxs[all_img_id[i]]
            y[nd] = 1
            scenes_id_alien[i] = y

        scenes_id_host = np.zeros((len(gt_host_idxs), n_data), dtype=np.uint32)
        for i in range(len(gt_host_idxs)):
            y = np.zeros(all_idxs.shape, dtype=np.uint32)
            nd = gt_host_idxs[all_img_id[i]]
            y[nd] = 1
            scenes_id_host[i] = y

        search_idxs = np.setdiff1d(all_idxs, query_idxs)
        search_idxs = np.sort(search_idxs)

        r_state = np.random.RandomState(7)

        for n in range(self.n_round):
            search_idxs = r_state.permutation(search_idxs)

        splited_search_idxs = np.array_split(search_idxs, self.n_batches, axis=0)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': scenes_id,
                  # 's_all_labels': [(scenes_id, "All")],
                  's_all_labels': [(scenes_id, "All"), (scenes_id_host, "Host"), (scenes_id_alien, "Alien")],
                  'all_idxs': all_idxs,
                  'distractor_idxs': distractor_idxs,
                  'search_idxs': splited_search_idxs,
                  # 'search_idxs': [np.sort(search_idxs)],
                  'all_search_idxs': np.sort(search_idxs),
                  'query_idxs': np.sort(query_idxs),
                  'sub_class': sub_class,
                  'donor_mask_fnames': donor_mask_fnames,
                  'all_img_id': all_img_id,
                  'hash_idx_probe': hash_idx_probe,
                  'hash_idx_world': hash_idx_world,
                  }

        return r_dict

    @property
    def meta_info(self):
        return self._build_meta(self.dataset_path, self.file_types)

    def meta_info_feats(self, output_path, file_types):
        return self._build_meta(output_path, file_types)

    def meta_info_images(self, output_path, file_types):
        return self._build_meta(output_path, file_types)
