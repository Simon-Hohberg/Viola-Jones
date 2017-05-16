from violajones import AdaBoost
from violajones.IntegralImage import IntegralImage
import os
import numpy as np
from violajones.HaarLikeFeature import FeatureType
from PIL import Image


def load_images(path, label):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            images.append(IntegralImage(os.path.join(path, _file), label))
    return images


def classify(classifiers, image):
    return 1 if sum([c[0].get_vote(image) * c[1] for c in classifiers]) >= 0 else -1


def reconstruct(classifiers):
    image = np.zeros((25,25))
    for (c, w) in classifiers:
        # map polarity: -1 -> 0, 1 -> 1
        polarity = pow(1 + c.polarity, 2)/4
        if c.type == FeatureType.TWO_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if y >= c.height/2:
                        sign = (sign + 1) % 2
                    image[c.top_left[1] + y, c.top_left[0] + x] += 255 * sign * w
        elif c.type == FeatureType.TWO_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x >= c.width/2:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 255 * sign * w
        elif c.type == FeatureType.THREE_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x % c.width/3 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 255 * sign * w
        elif c.type == FeatureType.THREE_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if x % c.height/3 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 255 * sign * w
        elif c.type == FeatureType.FOUR:
            sign = polarity
            for x in range(c.width):
                if x % c.width/2 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    if x % c.height/2 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 255 * sign * w
    result = Image.fromarray(image.astype(np.uint8)*255)
    return result


if __name__ == "__main__":
    
    # TODO: select optimal threshold for each feature
    # TODO: attentional cascading
    
    print('Loading faces..')
    faces = load_images('../trainingdata/faces', 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading non faces..')
    non_faces = load_images('../trainingdata/nonfaces', -1)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')
    
    T = 20
    # classifiers are haar like features
    # tuples (feature, weight)
    classifiers = AdaBoost.learn(faces, non_faces, T)
    
    print('Loading test faces..')
    faces = load_images('../trainingdata/faces/test', 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces = load_images('../trainingdata/nonfaces/test', -1)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')
    
    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    for image in faces + non_faces:
        result = classify(classifiers, image)
        if image.label == 1 and result == 1:
            correct_faces += 1
        if image.label == -1 and result == -1:
            correct_non_faces += 1
            
    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces))
        + '  (' + str((float(correct_faces)/len(faces))*100) + '%)\n  non-Faces: '
        + str(correct_non_faces) + '/' + str(len(non_faces)) + '  ('
        + str((float(correct_non_faces)/len(non_faces))*100) + '%)')
    
    # recon = reconstruct(classifiers)
    # recon.save('reconstruction.png')
