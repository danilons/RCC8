# coding: utf-8
import argparse
from rcc8nn.dataset.imdb import PascalVoc
from rcc8nn.nn.cnn import build_net

# def roi_nn():
#     net = {}
#     net['data'] = layers.InputLayer(shape=(None, 3, 32, 32))
#     # net['rois'] = layers.InputLayer(shape=(None, 4))
#
#     # conv1
#     net['conv1_1'] = Conv2DLayer(net['data'], num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, pad=1)
#     net['conv1_2'] = Conv2DLayer(net['conv1_1'], num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, pad=1)
#     net['conv1'] = lasagne.layers.concat((net['conv1_1'], net['conv1_2']), axis=1)
#     net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
#
#     # conv2
#     net['conv2_1'] = Conv2DLayer(net['pool1'], num_filters=128, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv2_2'] = Conv2DLayer(net['conv2_1'], num_filters=128, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv2'] = lasagne.layers.concat((net['conv2_1'], net['conv2_2']), axis=1)
#     net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(2, 2), stride=2)
#
#     # conv3
#     net['conv3_1'] = Conv2DLayer(net['pool2'], num_filters=256, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv3_2'] = Conv2DLayer(net['conv3_1'], num_filters=256, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv3_3'] = Conv2DLayer(net['conv3_2'], num_filters=256, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv3'] = lasagne.layers.concat((net['conv3_1'], net['conv3_2'], net['conv3_3']), axis=1)
#     net['pool3'] = MaxPool2DLayer(net['conv3'], pool_size=(2, 2))
#
#     # conv4
#     net['conv4_1'] = Conv2DLayer(net['pool3'], num_filters=512, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv4_2'] = Conv2DLayer(net['conv4_1'], num_filters=512, filter_size=(3, 3), nonlinearity=nonlinearities.rectify,pad=1)
#     net['conv4_3'] = Conv2DLayer(net['conv4_2'], num_filters=512, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv4'] = lasagne.layers.concat((net['conv4_1'], net['conv4_2'], net['conv4_3']), axis=1)
#     net['pool4'] = MaxPool2DLayer(net['conv4'], pool_size=(2, 2))
#
#     # conv5
#     net['conv5_1'] = Conv2DLayer(net['pool4'], num_filters=512, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv5_2'] = Conv2DLayer(net['conv5_1'], num_filters=512, filter_size=(3, 3), nonlinearity=nonlinearities.rectify, pad=1)
#     net['conv5_3'] = Conv2DLayer(net['conv5_2'], num_filters=512, filter_size=(3, 3), nonlinearity=nonlinearities.rectify,pad=1)
#     # net['conv5'] = lasagne.layers.concat((net['rois'], net['conv5_3']), axis=1)
#     net['pool5'] = MaxPool2DLayer(net['conv5_3'], pool_size=(2, 2))
#
#     # hidden layers
#     net['fc6'] = lasagne.layers.DenseLayer(net['pool5'], num_units=4096, nonlinearity=nonlinearities.rectify)
#     net['drop6'] = lasagne.layers.DropoutLayer(net['fc6'], p=0.5)
#     net['fc7'] = lasagne.layers.DenseLayer(net['drop6'], num_units=4096, nonlinearity=nonlinearities.rectify)
#     net['drop7'] = lasagne.layers.DropoutLayer(net['fc7'], p=0.5)
#
#     # scores
#     net['cls_score'] = lasagne.layers.DenseLayer(net['drop7'], num_units=21, nonlinearity=None)
#     net['bbox_pred'] = lasagne.layers.DenseLayer(net['drop7'], num_units=84, nonlinearity=None)
#     net['cls_proba'] = lasagne.layers.DenseLayer(net['cls_score'], num_units=21, nonlinearity=lasagne.nonlinearities.softmax)\
#
#     return net
#


def training(dataset, epochs=None):
    """

    :param dataset:
    :param epochs:
    :return:
    """
    net = build_net()
    net.fit(dataset['train']['X1'], dataset['train']['Y'], epochs=None)
    return net

def parse_args():
    """
    Parse input

    :return: obj of parsed args
    """
    parser = argparse.ArgumentParser(description='Train an one layer NN network')
    parser.add_argument('--dset', help='Dataset (default PASCAL VOC 2007)', dest='dset')
    parser.add_argument('--path', help='Local path where dataset is located', dest='path')
    parser.add_argument('--model', help='Local filename to save models', dest='model')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # create a dataset handle
    dataset = PascalVoc(path=args.path)

    # run all the stuff
    net = training(dataset)

    # save net
    net.save_params_to(args.model)