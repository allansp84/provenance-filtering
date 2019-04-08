# -*- coding: utf-8 -*-

from provenancefiltering.icip17.datasets.oxford100k import Oxford100k


class Unicamp100k(Oxford100k):

    def __init__(self, dataset_path, output_path='./working', file_types=('jpg', 'png'),
                 groundtruth_path='', n_round=1, n_batches=4, query_id='', n_samples=0):

        super(Unicamp100k, self).__init__(dataset_path, output_path, file_types,
                                          groundtruth_path, n_round, n_batches, query_id, n_samples)
