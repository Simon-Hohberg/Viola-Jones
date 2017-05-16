import unittest
from violajones.HaarLikeFeature import HaarLikeFeature
from violajones.HaarLikeFeature import FeatureType
from violajones.IntegralImage import IntegralImage


class HaarLikeFeatureTest(unittest.TestCase):

    def setUp(self):
        self.intImage = IntegralImage('../trainingdata/faces/faces0001.png', 0)
    
    def tearDown(self):
        pass

    def test_two_vertical(self):
        feature = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0,0), 24, 24, 100000, 1);
        left_area = self.intImage.get_area_sum((0,0), (24, 12))
        right_area = self.intImage.get_area_sum((0,12), (24,24))
        expected = 1 if feature.threshold * feature.polarity > left_area - right_area else 0
        assert feature.get_vote(self.intImage) == expected
        
    def test_two_vertical_fail(self):
        feature = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0,0), 24, 24, 100000, 1);
        left_area = self.intImage.get_area_sum((0,0), (24, 12))
        right_area = self.intImage.get_area_sum((0,12), (24,24))
        expected = 1 if feature.threshold * -1 > left_area - right_area else 0
        assert feature.get_vote(self.intImage) != expected 

    def test_two_horizontal(self):
        feature = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0,0), 24, 24, 100000, 1);
        left_area = self.intImage.get_area_sum((0,0), (12, 24))
        right_area = self.intImage.get_area_sum((12,0), (24,24))
        expected = 1 if feature.threshold * feature.polarity > left_area - right_area else 0
        assert feature.get_vote(self.intImage) == expected
        
    def test_three_horizontal(self):
        feature = HaarLikeFeature(FeatureType.THREE_HORIZONTAL, (0,0), 24, 24, 100000, 1);
        left_area = self.intImage.get_area_sum((0,0), (8, 24))
        middle_area = self.intImage.get_area_sum((8,0), (16,24))
        right_area = self.intImage.get_area_sum((16,0), (24,24))
        expected = 1 if feature.threshold * feature.polarity > left_area - middle_area + right_area else 0
        assert feature.get_vote(self.intImage) == expected
        
    def test_three_vertical(self):
        feature = HaarLikeFeature(FeatureType.THREE_VERTICAL, (0,0), 24, 24, 100000, 1);
        left_area = self.intImage.get_area_sum((0,0), (24, 8))
        middle_area = self.intImage.get_area_sum((0,8), (24,16))
        right_area = self.intImage.get_area_sum((0,16), (24,24))
        expected = 1 if feature.threshold * feature.polarity > left_area - middle_area + right_area else 0
        assert feature.get_vote(self.intImage) == expected
        
    def test_four(self):
        feature = HaarLikeFeature(FeatureType.THREE_HORIZONTAL, (0,0), 24, 24, 100000, 1);
        top_left_area = self.intImage.get_area_sum((0,0), (12, 12))
        top_right_area = self.intImage.get_area_sum((12,0), (24,12))
        bottom_left_area = self.intImage.get_area_sum((0,12), (12,24))
        bottom_right_area = self.intImage.get_area_sum((12,12), (24,24))
        expected = 1 if feature.threshold * feature.polarity > top_left_area - top_right_area - bottom_left_area + bottom_right_area else 0
        assert feature.get_vote(self.intImage) == expected


if __name__ == "__main__":
    unittest.main()
