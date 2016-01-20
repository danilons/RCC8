# coding: utf-8
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.layers import Conv2DLayer as Conv2DLayer
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer
from nolearn.lasagne import NeuralNet


def float32(k):
    return np.cast['float32'](k)


def build_net():
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            # ('rois', layers.InputLayer),
            ('conv1_1', Conv2DLayer),
            ('conv1_2', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('conv2_1', Conv2DLayer),
            ('conv2_2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('conv3_1', Conv2DLayer),
            ('conv3_2', Conv2DLayer),
            ('conv3_3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            ('conv4_1', Conv2DLayer),
            ('conv4_2', Conv2DLayer),
            ('conv4_3', Conv2DLayer),
            ('pool4', MaxPool2DLayer),
            ('conv5_1', Conv2DLayer),
            ('conv5_2', Conv2DLayer),
            ('conv5_3', Conv2DLayer),
            ('pool5', MaxPool2DLayer),
            ('fc6',  layers.DenseLayer),
            ('drop6', layers.DropoutLayer),
            ('fc7', layers.DenseLayer),
            ('drop7', layers.DropoutLayer),
            ('cls_score', layers.DenseLayer),
            ('bbox_pred', layers.DenseLayer),
            ('cls_prob', layers.DenseLayer)
        ],
        input_shape=(None, 3, 32, 32),
        # conv1
        conv1_1_num_filters=64, conv1_1_filter_size=(3, 3), conv1_1_pad=1,
        conv1_2_num_filters=64, conv1_2_filter_size=(3, 3), conv1_2_pad=2,
        pool1_pool_size=(2, 2),

        # conv2
        conv2_1_num_filters=128, conv2_1_filter_size=(3, 3), conv2_1_pad=1,
        conv2_2_num_filters=128, conv2_2_filter_size=(3, 3), conv2_2_pad=2,
        pool2_pool_size=(2, 2),

        # conv3
        conv3_1_num_filters=256, conv3_1_filter_size=(3, 3), conv3_1_pad=1,
        conv3_2_num_filters=256, conv3_2_filter_size=(3, 3), conv3_2_pad=2,
        conv3_3_num_filters=256, conv3_3_filter_size=(3, 3), conv3_3_pad=2,
        pool3_pool_size=(2, 2),

        # conv4
        conv4_1_num_filters=512, conv4_1_filter_size=(3, 3), conv4_1_pad=1,
        conv4_2_num_filters=512, conv4_2_filter_size=(3, 3), conv4_2_pad=2,
        conv4_3_num_filters=512, conv4_3_filter_size=(3, 3), conv4_3_pad=2,
        pool4_pool_size=(2, 2),

        # conv5
        conv5_1_num_filters=512, conv5_1_filter_size=(3, 3), conv5_1_pad=1,
        conv5_2_num_filters=512, conv5_2_filter_size=(3, 3), conv5_2_pad=2,
        conv5_3_num_filters=512, conv5_3_filter_size=(3, 3), conv5_3_pad=2,
        pool5_pool_size=(2, 2),

        # inner product
        fc6_num_units=4096, fc6_nonlinearity=lasagne.nonlinearities.rectify,
        drop6_p=0.5,

        fc7_num_units=4096, fc7_nonlinearity=lasagne.nonlinearities.rectify,
        drop7_p=0.5,

        # probabilities
        cls_score_num_units=21,
        bbox_pred_num_units=84,
        cls_prob_num_units=21, cls_prob_nonlinearity=lasagne.nonlinearities.softmax,

        # solver
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),

        max_epochs=10,
        verbose=1,
    )
    return net
