import builtins as __builtin__

import pytest
from mock import mock_open, patch

from RCC8NN.rcc8nn.dataset import imdb


@pytest.fixture
def annot():
    return """<annotation><object><name>chair</name><pose>Rear</pose><truncated>0</truncated>
            <difficult>0</difficult><bndbox><xmin>263</xmin><ymin>211</ymin><xmax>324</xmax>
            <ymax>339</ymax></bndbox></object><object><name>chair</name><pose>Unspecified</pose>
            <truncated>0</truncated><difficult>0</difficult>
            <bndbox><xmin>165</xmin><ymin>264</ymin><xmax>253</xmax>
            <ymax>372</ymax></bndbox></object></annotation>"""

@pytest.fixture
def dset():
    return imdb.Dataset(path='')


def test_annotation_load(annot, dset):
    m = mock_open()
    with patch.object(__builtin__, 'open', mock_open(read_data=annot)):
        with open('annotated_fixture.xml', 'w') as fp:
            fp.read()

            assert dset.load_annotation('0')
            assert len(dset.load_annotation('0')) == 4  # keys {boxes, classes, overlaps, flipped}
            assert dset.load_annotation('0')['boxes'].shape[1] == 4
