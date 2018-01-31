import numpy as np
from violajones.HaarLikeFeature import FeatureType
from functools import partial
import os
from PIL import Image


def ensemble_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0


def ensemble_vote_all(int_imgs, classifiers):
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: List of assigned labels, 1 if image was classified positively, else
    0
    :rtype: list[int]
    """
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


def reconstruct(classifiers, img_size):
    """
    Creates an image by putting all given classifiers on top of each other
    producing an archetype of the learned class of object.
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :param img_size: Tuple of width and height
    :type img_size: (int, int)
    :return: Reconstructed image
    :rtype: numpy.ndarray
    """
    image = np.zeros(img_size)
    for c in classifiers:
        # map polarity: -1 -> 0, 1 -> 1
        polarity = pow(1 + c.polarity, 2)/4
        if c.type == FeatureType.TWO_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if y >= c.height/2:
                        sign = (sign + 1) % 2
                    image[c.top_left[1] + y, c.top_left[0] + x] += 1 * sign * c.weight
        elif c.type == FeatureType.TWO_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x >= c.width/2:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x % c.width/3 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if x % c.height/3 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.FOUR:
            sign = polarity
            for x in range(c.width):
                if x % c.width/2 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    if x % c.height/2 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
    image -= image.min()
    image /= image.max()
    return image


def load_images(path):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            img_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            if img_arr.max() > 0:
                img_arr /= img_arr.max()
            images.append(img_arr)
    return images
