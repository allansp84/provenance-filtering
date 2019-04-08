# -*- coding: utf-8 -*-

import os
from abc import ABCMeta
from abc import abstractmethod


class Dataset(metaclass=ABCMeta):

    def __init__(self, dataset_path, output_path, file_types):

        self.__dataset_path = ""
        self.__output_path = ""
        self.__file_types = ""

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.file_types = file_types

        self.__meta_info = None
        self.meta_was_built = False

    @property
    def dataset_path(self):
        return self.__dataset_path

    @dataset_path.setter
    def dataset_path(self, path):
        self.__dataset_path = os.path.abspath(path)

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        self.__output_path = os.path.abspath(path)

    @property
    def file_types(self):
        return self.__file_types

    @file_types.setter
    def file_types(self, file_types):
        self.__file_types = file_types

    @abstractmethod
    def _build_meta(self, in_path, file_types):
        pass

    @staticmethod
    def _list_dirs(roo_tpath, file_types):
        folders = []

        for root, dirs, files in os.walk(roo_tpath, followlinks=True):
            for f in files:
                if os.path.splitext(f)[1].replace(".", "").lower() in file_types:
                    folders += [os.path.relpath(root, roo_tpath)]
                    break

        return folders

    @abstractmethod
    def _build_meta(self, in_path, file_types):
        pass

    @property
    def meta_info(self):
        """ Metadata of the images.

        Returns:
            dict: A dictionary contaning the metadata.
        """

        if not self.meta_was_built:
            self.__meta_info = self._build_meta(self.dataset_path, self.file_types)
            self.meta_was_built = True

        return self.__meta_info
