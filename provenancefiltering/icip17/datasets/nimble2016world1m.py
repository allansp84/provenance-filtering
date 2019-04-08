# -*- coding: utf-8 -*-

from provenancefiltering.icip17.datasets.nimble2016 import Nimble2016


class Nimble2016World1M(Nimble2016):

    def __init__(self, dataset_path, output_path='./working', file_types=('jpg', 'png', 'gif'),
                 n_round=1, n_batches=4):

        super(Nimble2016World1M, self).__init__(dataset_path, output_path, file_types, n_round, n_batches)
