import numpy as np
from HaarLikeFeature import FeatureType
from HaarLikeFeature import HaarLikeFeature

class AdaBoost(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
def learn(positives, negatives, T):
    
    feature = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0,0), 1, 1, 3, 1)
    
    # construct initial weights
    weights_pos = np.ones((T, 1)) * 1/(2 * len(positives))
    weights_neg = np.ones((T, 1)) * 1/(2 * len(negatives))
    weights_pos = np.hstack((weights_pos, np.zeros(len(positives)-1)))
    weights_neg = np.hstack((weights_neg, np.zeros(len(negatives)-1)))
    
    weights = np.hstack((weights_pos, weights_neg))
    
    for i in range(T):
        
        # normalize weights
        weights *= 1/sum(weights)
        
        # select best weak classifier
        