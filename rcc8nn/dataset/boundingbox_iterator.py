# coding: utf-8

from nolearn.lasagne import BatchIterator


class BoundingBoxBatch(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(BoundingBoxBatch, self).transform(Xb, yb)
