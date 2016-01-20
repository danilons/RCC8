# coding: utf-8
import os
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import skimage.io
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder


class Dataset(object):
    """
    Dataset interface

    adapted from fast R-CNN code
    """

    def __init__(self, path, classes=[]):
        """
        Constructor

        :param path: dataset location
        :param dset: dataset type (train, val, test)
        :return:
        """
        self.path = path
        self._image_ext = '.jpg'
        self.n_classes = len(classes)
        self.encoder = LabelEncoder().fit(classes)
        # load image set
        self.train = self._load_image_set('train')
        self.valid = self._load_image_set('val')

    def _load_image_set(self, dset_type):
        """
        load images containing a specific set (train, val, test)

        :return:
        """
        filename = os.path.join(self.path, 'ImageSets', 'Main', dset_type + '.txt')
        with open(filename, 'r') as fp:
            image_index = [fn.strip() for fn in fp]
        return image_index or []

    def load_annotation(self, index, zero_based=False):
        """
        load bounding boxes annotation

        :param index ID of image
        :param zero_based  coordinates starts with 1 (default) or not
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

    def batch(self, train_set, batch_size=20):
        """
        Return a minibatch of images

        :param batch_size: size of batch
        :return:
        """
        for index in train_set[:batch_size]:
            filename = os.path.join(self.path, 'JPEGImages', index + self._image_ext)
            annotations = self.load_annotation(index)
            image = skimage.io.imread(filename)
            yield resize(image, (32, 32), mode='nearest').T.astype(np.float32), \
                  annotations['boxes'], \
                  annotations['classes'].argmax().astype(np.int32)


class PascalVoc(Dataset):

    classes_ = ('__background__', # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, path=None, batch_size=20):
        super(PascalVoc, self).__init__(path=path, classes=self.classes_)
        self.db = {}
        self.db['train'] = self._create_db(self.train, batch_size=batch_size)
        self.db['valid'] = self._create_db(self.valid, batch_size=batch_size)

    def _create_db(self, dset_type, batch_size):
        """
        Create a dict containing X1, X2, Y for a given training set (train, valid)

        :param dset_type:
        :return:
        """
        trainset = {}
        x1, x2, y = zip(*[data for data in self.batch(dset_type, batch_size=batch_size)])
        trainset['X1'] = np.array(x1)
        trainset['X2'] = np.array(x2)
        trainset['Y'] = np.array(y)
        return trainset

    def __getitem__(self, item):
        return self.db[item]






