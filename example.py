import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.Utils as utils

if __name__ == "__main__":
    pos_training_path = 'trainingdata/faces'
    neg_training_path = 'trainingdata/nonfaces'
    pos_testing_path = 'trainingdata/faces/test'
    neg_testing_path = 'trainingdata/nonfaces/test'

    num_classifiers = 2
    # For performance reasons restricting feature size
    min_feature_height = 8
    max_feature_height = 10
    min_feature_width = 8
    max_feature_width = 10

    print('Loading faces..')
    faces_training = utils.load_images(pos_training_path)
    faces_ii_training = list(map(ii.to_integral_image, faces_training))
    print('..done. ' + str(len(faces_training)) + ' faces loaded.\n\nLoading non faces..')
    non_faces_training = utils.load_images(neg_training_path)
    non_faces_ii_training = list(map(ii.to_integral_image, non_faces_training))
    print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')

    # classifiers are haar like features
    classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)

    print('Loading test faces..')
    faces_testing = utils.load_images(pos_testing_path)
    faces_ii_testing = list(map(ii.to_integral_image, faces_testing))
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = utils.load_images(neg_testing_path)
    non_faces_ii_testing = list(map(ii.to_integral_image, non_faces_testing))
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(utils.ensemble_vote_all(faces_ii_testing, classifiers))
    correct_non_faces = len(non_faces_testing) - sum(utils.ensemble_vote_all(non_faces_ii_testing, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
          + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')

    # Just for fun: putting all haar-like features over each other generates a face-like image
    recon = utils.reconstruct(classifiers, faces_testing[0].shape)
    recon.save('reconstruction.png')
