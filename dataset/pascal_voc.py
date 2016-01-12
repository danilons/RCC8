# coding: utf-8
import os
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse

from sklearn.preprocessing import LabelEncoder


class Dataset(object):
    """
    Dataset interface

    adapted from fast R-CNN code
    """

    def __init__(self, path, classes=None):
        """
        Constructor

        :param path: dataset location
        :return:
        """
        self.path = path
        # read default pascal voc class
        classes = classes or ('__background__', # always index 0
                             'aeroplane', 'bicycle', 'bird', 'boat',
                             'bottle', 'bus', 'car', 'cat', 'chair',
                             'cow', 'diningtable', 'dog', 'horse',
                             'motorbike', 'person', 'pottedplant',
                             'sheep', 'sofa', 'train', 'tvmonitor')
        self.n_classes = len(classes)

        self.encoder = LabelEncoder().fit(classes)

    def load_annotation(self, index, zero_based=False):
        """
        load bounding boxes annotation

        :param index ID of image
        :return: list of [xmin, ymin, xmax, ymax] location
        """
        filename = os.path.join(self.path, 'Annotations', index + '.xml')

        with open(filename, 'r') as fp:
            content = minidom.parseString(fp.read())

        if not content:
            return None

        objects = content.getElementsByTagName('object')
        n_objs = len(objects)

        boxes = np.zeros(shape=(n_objs, 4), dtype=np.uint16)
        classes = np.zeros(shape=(n_objs, 1), dtype=np.int32)
        overlaps = np.zeros((n_objs, self.n_classes), dtype=np.float32)

        shift = 0 if zero_based else 1

        for idx, obj in enumerate(objects):
            xmin = float(self.get_data_from_tag(obj, 'xmin')) - shift
            ymin = float(self.get_data_from_tag(obj, 'ymin')) - shift
            xmax = float(self.get_data_from_tag(obj, 'xmax')) - shift
            ymax = float(self.get_data_from_tag(obj, 'ymax')) - shift

            cls_name = self.get_data_from_tag(obj, "name")
            clazz = self.encoder.transform(cls_name)

            # update
            boxes[idx, :] = np.array([xmin, ymin, xmax, ymax])
            classes[idx, 0] = clazz
            overlaps[idx, :] = 1.0

        return {'boxes': boxes, 'classes': classes, 'overlaps':  scipy.sparse.csr_matrix(overlaps), 'flipped': False}

    def get_data_from_tag(self, node, tag):
        """
        Interface to read info from a specific tag

        :param node:
        :param tag:
        :return: content of xml tag
        """
        return node.getElementsByTagName(tag)[0].childNodes[0].data