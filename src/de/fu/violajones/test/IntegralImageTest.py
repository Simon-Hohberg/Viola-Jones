import unittest
from de.fu.violajones.IntegralImage import IntegralImage
import numpy as np


class IntegralImageTest(unittest.TestCase):


    def setUp(self):
        self.intImage = IntegralImage('../../../../../../trainingdata/faces/faces0001.png', 0)

    def tearDown(self):
        pass


    def test_integral_calculation(self):
        assert self.intImage.integral[1, 1] == self.intImage.original[0, 0]
        assert self.intImage.integral[-1, 1] == np.sum(self.intImage.original[:, 0])
        assert self.intImage.integral[1, -1] == np.sum(self.intImage.original[0, :])
        assert self.intImage.integral[-1, -1] == np.sum(self.intImage.original)
        
    def test_area_sum(self):
        assert self.intImage.get_area_sum((0,0), (1,1)) == self.intImage.original[0, 0]
        assert self.intImage.get_area_sum((0,0), (-1,-1)) == np.sum(self.intImage.original)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()