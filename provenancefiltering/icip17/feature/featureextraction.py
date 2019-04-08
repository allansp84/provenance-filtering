# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np

from PIL import Image

from provenancefiltering.icip17.feature.detection.extraction import local_feature_detection_and_description
from provenancefiltering.icip17.utils import safe_create_dir, DESCRIPTOR_SIZE, DESCRIPTOR_TYPE
from provenancefiltering.icip17.feature.detection.keypoints import keypoints_to_array

verbose = True


class FeatureExtraction(object):

    def __init__(self, input_fnames, output_fnames, detector, descriptor, limit, resize_img,
                 force_written=False, use_map=False, output_fnames_maps='', img=None, mask_img=None,
                 return_fv=False, default_params=True):

        self.input_fnames = input_fnames
        self.output_fnames = output_fnames
        self.force_written = force_written
        self.use_map = use_map

        self.detector = detector
        self.descriptor = descriptor
        self.limit = limit
        self.resize_img = resize_img
        self.output_fnames_maps = output_fnames_maps

        self.img = img
        self.mask_img = mask_img
        self.return_fv = return_fv
        self.default_params = default_params

    def load_images(self):

        try:
            if 'gif' in os.path.splitext(self.input_fnames)[1]:
                pil_img = Image.open(self.input_fnames).convert("RGB")
                img = np.array(pil_img.getdata()).reshape(pil_img.size[1], pil_img.size[0], 3)[:, :, ::-1]
            else:
                img = cv2.imread(self.input_fnames, cv2.IMREAD_COLOR)

            if self.resize_img:
                dec_factor = 0.75
                n_rows, n_cols = img.shape[:2]
                img = cv2.resize(img, (int(n_cols * dec_factor), int(n_rows * dec_factor)))

            img = np.array(img, dtype=np.uint8)

        except Exception:
            raise(Exception, 'Error: Can not read the image %s' % self.input_fnames)

        return img

    def extract_features(self):

        key_points, feature_vectors, det_t, dsc_t = local_feature_detection_and_description(self.input_fnames, self.detector,
                                                                                            self.descriptor,
                                                                                            kmax=self.limit,
                                                                                            img=self.img,
                                                                                            mask=self.mask_img,
                                                                                            default_params=self.default_params,
                                                                                            )

        if len(feature_vectors) == 0:
            feature_vectors = np.zeros((1, DESCRIPTOR_SIZE[self.descriptor]), dtype=DESCRIPTOR_TYPE[self.descriptor])
            key_points = np.zeros((1, 7), dtype=DESCRIPTOR_TYPE[self.descriptor])

        if isinstance(key_points[0], cv2.KeyPoint().__class__):
            key_points = keypoints_to_array(key_points)
        else:
            key_points = np.array(key_points, dtype=np.float32)

        run_times = [det_t, dsc_t]

        return key_points, feature_vectors, run_times

    def save_features(self, key_points, feature_vectors, run_times):

        if verbose:
            print("    saving {0} feature extracted from {1}".format(self.descriptor, self.output_fnames))
            sys.stdout.flush()

        safe_create_dir(os.path.dirname(self.output_fnames))

        key_point_path = "{0}.kp".format(os.path.splitext(self.output_fnames)[0])
        np.savetxt(key_point_path, key_points, fmt='%.4f')
        np.save(self.output_fnames, feature_vectors)

        if verbose:
            print("    Done! (Detection: {0:0.4f}s / Description: {1:0.4f}s)".format(run_times[0], run_times[1]))
            print("    Outfile is: {0:s}\n".format(self.output_fnames))
            sys.stdout.flush()

    def run(self):

        if not self.force_written and os.path.exists(self.output_fnames):
            if verbose:
                print("-- File already exits {0}".format(self.output_fnames))

            if self.return_fv:
                return [[], np.load(self.output_fnames)]

            return True

        if self.img is None:
            self.img = self.load_images()

        key_points, feature_vectors, run_times = self.extract_features()

        if self.return_fv:
            return [feature_vectors]

        self.save_features(key_points, feature_vectors, run_times)

        return True
