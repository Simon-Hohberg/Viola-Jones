import numpy as np
from de.fu.violajones.HaarLikeFeature import HaarLikeFeature
from de.fu.violajones.HaarLikeFeature import FeatureTypes
import sys

LOADING_BAR_LENGTH = 50

class AdaBoost(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
def learn(positives, negatives, T):
    
    # construct initial weights
    pos_weight = 1. / (2 * len(positives))
    neg_weight = 1. / (2 * len(negatives))
    for p in positives:
        p.set_weight(pos_weight)
    for n in negatives:
        n.set_weight(neg_weight)
    
    # create column vector
    images = np.hstack((positives, negatives))
    
    print('Creating haar like features..')
    features = []
    for f in FeatureTypes:
        for width in range(f[0],25,f[0]):
            for height in range(f[1],25,f[1]):
                for x in range(25-width):
                    for y in range(25-height):
                        features.append(HaarLikeFeature(f, (x,y), width, height, 0, 1))
    print('..done.\n' + str(len(features)) + ' features created.\n')
    
    print('Calculating scores for features..')
    sys.stdout.write('[' + ' '*LOADING_BAR_LENGTH + ']\r')
    sys.stdout.flush()
    # dictionary of feature -> list of vote for each image: matrix[image, weight, vote])
    votes = dict()
    i = 0
    for feature in features:
        # calculate score for each image, also associate the image
        feature_votes = np.array(list(map(lambda im: [im, feature.get_vote(im)], images)))
        votes[feature] = feature_votes
        i += 1
        if i % 1000 == 0:
            break   # TODO: remove
            progres = int(((i + 1) * LOADING_BAR_LENGTH) / len(features))
            sys.stdout.write('[' + '='*progres + ' '*(LOADING_BAR_LENGTH-progres) + ']\r')
            sys.stdout.flush()
    print('\n')
    
    
    # select classifiers
    
    classifiers = []
    
    print('Selecting classifiers..')
    sys.stdout.write('[' + ' '*LOADING_BAR_LENGTH + ']\r')
    sys.stdout.flush()
    for i in range(T):
        
        classification_errors = dict()

        # normalize weights
        norm_factor = 1. / sum(map(lambda im: im.weight, images))
        for image in images:
            image.set_weight(image.weight * norm_factor)

        # select best weak classifier
        for feature, feature_votes in votes.items():
            # calculate error
            error = sum(map(lambda im, vote: im.weight if im.label != vote else 0, feature_votes[:,0], feature_votes[:,1]))
            # map error -> feature, use error as key to select feature with
            # smallest error later
            classification_errors[error] = feature
        
        # get best feature, i.e. with smallest error
        errors = list(classification_errors.keys())
        best_error = errors[np.argmin(errors)]
        feature = classification_errors[best_error]
        feature_weight = 0.5 * np.log((1-best_error)/best_error)
        
        classifiers.append((feature, feature_weight))
        
        # update image weights
        best_feature_votes = votes[feature]
        for feature_vote in best_feature_votes:
            im = feature_vote[0]
            vote = feature_vote[1]
            if im.label != vote:
                im.set_weight(im.weight * np.sqrt((1-best_error)/best_error))
            else:
                im.set_weight(im.weight * np.sqrt(best_error/(1-best_error)))
        
        # remove feature (a feature can't be selected twice)
        votes.pop(feature)
        
        progres = int(((i + 1) * LOADING_BAR_LENGTH) / T)
        sys.stdout.write('[' + '='*progres + ' '*(LOADING_BAR_LENGTH-progres) + ']\r')
        sys.stdout.flush()
    print('\n')
    
    return classifiers
        