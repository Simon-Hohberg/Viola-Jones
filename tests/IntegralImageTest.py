import unittest
import violajones.IntegralImage as ii
from PIL import Image
import numpy as np


class IntegralImageTest(unittest.TestCase):

    def setUp(self):
        self.orig_img = np.array(Image.open('../trainingdata/faces/faces0001.png'), dtype=np.float64)
        self.int_img = ii.to_integral_image(self.orig_img)

    def tearDown(self):
        pass

    def test_integral_calculation(self):
        assert self.int_img[1, 1] == self.orig_img[0, 0]
        assert self.int_img[-1, 1] == np.sum(self.orig_img[:, 0])
        assert self.int_img[1, -1] == np.sum(self.orig_img[0, :])
        assert self.int_img[-1, -1] == np.sum(self.orig_img)
        
    def test_area_sum(self):
        assert ii.sum_region(self.int_img, (0, 0), (1, 1)) == self.orig_img[0, 0]
        assert ii.sum_region(self.int_img, (0, 0), (-1, -1)) == np.sum(self.orig_img)

if __name__ == "__main__":
    unittest.main()