# -*- coding: utf-8 -*-

from provenancefiltering.icip17.datasets.nimble2017 import Nimble2017


class Nimble2017World1M(Nimble2017):

    def __init__(self, dataset_path, output_path='./working', file_types=('jpg', 'JPG', 'png', 'PNG', 'gif'),
                 n_round=1, n_batches=4):

        super(Nimble2017World1M, self).__init__(dataset_path, output_path, file_types, n_round, n_batches)
