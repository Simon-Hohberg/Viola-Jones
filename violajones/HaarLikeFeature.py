import IntegralImage as ii


def enum(**enums):
    return type('Enum', (), enums)

FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


class HaarLikeFeature(object):
    """
    Class representing a haar-like feature.
    """

    def __init__(self, feature_type, position, width, height, threshold, polarity):
        """
        Creates a new haar-like feature.
        @param feature_type: see FeatureType enum
        @param position: top left corner where the feature begins (tuple)
        @param width: width of the feature
        @param height: height of the feature
        @param threshold: feature threshold
        @param polarity: polarity of the feature -1 or 1
        """
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = 1
    
    def get_score(self, int_img):
        """
        Get score for given integral image array.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: Score for given feature
        :rtype: int
        """
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width, self.top_left[1] + self.height / 2))
            second = ii.sum_region(int_img, (self.top_left[0], self.top_left[1] + self.height / 2), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width / 2, self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (self.top_left[0] + self.width / 2, self.top_left[1]), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width / 3, self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (self.top_left[0] + self.width / 3, self.top_left[1]), (self.top_left[0] + 2 * self.width / 3, self.top_left[1] + self.height))
            third = ii.sum_region(int_img, (self.top_left[0] + 2 * self.width / 3, self.top_left[1]), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = ii.sum_region(int_img, self.top_left, (self.bottom_right[0], self.top_left[1] + self.height / 3))
            second = ii.sum_region(int_img, (self.top_left[0], self.top_left[1] + self.height / 3), (self.bottom_right[0], self.top_left[1] + 2 * self.height / 3))
            third = ii.sum_region(int_img, (self.top_left[0], self.top_left[1] + 2 * self.height / 3), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width / 2, self.top_left[1] + self.height / 2))
            # top right area
            second = ii.sum_region(int_img, (self.top_left[0] + self.width / 2, self.top_left[1]), (self.bottom_right[0], self.top_left[1] + self.height / 2))
            # bottom left area
            third = ii.sum_region(int_img, (self.top_left[0], self.top_left[1] + self.height / 2), (self.top_left[0] + self.width / 2, self.bottom_right[1]))
            # bottom right area
            fourth = ii.sum_region(int_img, (self.top_left[0] + self.width / 2, self.top_left[1] + self.height / 2), self.bottom_right)
            score = first - second - third + fourth
        return score
    
    def get_vote(self, int_img):
        """
        Get vote of this feature for given integral image.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: 1 iff this feature votes positively, otherwise -1
        :rtype: int
        """
        score = self.get_score(int_img)
        return self.weight * (1 if score < self.polarity * self.threshold else -1)
