# import modules
import numpy as np
import librosa
import CNN_logic as cnn
from keras.utils import np_utils
import time
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix

def main(random_seed=None, visualize_label=False, ann_model=cnn.baseline_model_96):


    # determine the random seed so that results are reproducible
    # random_seed = 11 # also determines which shuffled index to use
    np.random.seed(random_seed)

    # load and organize data
    # if data has not been converted yet change this to false
    data_converted = True
    if data_converted == False:
        # load all the data
        X, SR, T = cnn.load_original_data()
        # data format:
        #       x: 1d numpy array
        #       t: 1d numpy array with numsic genre names (numeric arrays or multinomial vector?)

        # convert the data into mel-scale spectrogram
        st = time.time()
        newX = cnn.batch_mel_spectrogram(X, SR)
        print(time.time() - st)

        # save the data into npz
        np.savez_compressed("audio_sr_label.npz", X=newX, SR=SR, T=T)

    else:
        st = time.time()
        data = np.load("audio_sr_label.npz")
        X = data["X"]
        SR = data["SR"]
        T = data["T"]
        loading_time = time.time() - st
        print("Loading takes %f seconds." % (loading_time))


    # Use log transformation to preserve the order but shrink the range
    X = np.log(X + 1)
    X = X[:, :, :, np.newaxis]  # image channel should be the last dimension, check by using print K.image_data_format()



    # convert string type labels to vectors
    genres = np.unique(T)
    genres_dict = dict([[label, value] for value, label in enumerate(genres)])
    T_numeric = np.asarray([genres_dict[label] for label in T])
    T_vectorized = np_utils.to_categorical(T_numeric)


    # split data into training, cross-validation,  testing data
    # following is used to generate random see used to split the data into different sets
    split_idxes = np.asarray([0, 0.5, 0.7, 1])
    training_idxes_list, validation_idxes_list, testing_idxes_list = [], [], []
    for idx in range(30):
        training_idxes, validation_idxes, testing_idxes = cnn.split_data(T, split_idxes)
        training_idxes_list.append(training_idxes)
        validation_idxes_list.append(validation_idxes)
        testing_idxes_list.append(testing_idxes)
    
    # training_idxes_list = np.asarray(training_idxes_list)
    # validation_idxes_list = np.asarray(validation_idxes_list)
    # testing_idxes_list = np.asarray(testing_idxes_list)
    
    np.savez_compressed("shuffled_idx_list.npz", training_idxes_list=training_idxes_list,
                        validation_idxes_list=validation_idxes_list, testing_idxes_list=testing_idxes_list)


    ## load one fixed data shuffling indexes
    idxes_list = np.load("shuffled_idx_list.npz")
    training_idxes = idxes_list["training_idxes_list"][random_seed]
    validation_idxes = idxes_list["validation_idxes_list"][random_seed]
    testing_idxes = idxes_list["testing_idxes_list"][random_seed]

    training_X = X[training_idxes]
    validation_X = X[validation_idxes]
    testing_X = X[testing_idxes]

    training_T = T_vectorized[training_idxes]
    validation_T = T_vectorized[validation_idxes]
    testing_T = T_vectorized[testing_idxes]


    print("Starting Training")
    MGCNN = cnn.Music_Genre_CNN(ann_model)
    # training the model
    training_flag = True
    max_iterations = 1
    while training_flag and max_iterations >= 0:
        validation_accuracies = MGCNN.train_model(training_X, training_T, cv=True,
                                                    validation_spectrograms=validation_X,
                                                    validation_labels=validation_T)

        diff = np.mean(validation_accuracies[-10:]) - np.mean(validation_accuracies[:10])
        if np.abs(diff) < 0.01:
            training_flag = False
        max_iterations -= 1


    test_accuracy, confusion_data = MGCNN.test_model(testing_X, testing_T)
    print("\n ****** The final test accuracy is %f. ******\n" % (test_accuracy))

    import matplotlib.pyplot as plt
    cm = confusion_matrix(confusion_data[:, 1], confusion_data[:, 0]) / (len(testing_T) * 1.0 / len(genres))

    plt.matshow(cm)
    plt.show()


main(random_seed=20, visualize_label=True, ann_model=cnn.baseline_model_96)