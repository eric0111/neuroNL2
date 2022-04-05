import numpy as np
from nilearn import datasets
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

#nilearn - neuroimaging tailored library
from nilearn.input_data import NiftiMapsMasker
from nilearn import plotting

#sklearn - basic ML tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn import metrics

#keras - for NN models
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers import LSTM, GRU
from keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import utils
from sklearn.metrics import roc_curve
from keras.utils.vis_utils import plot_model
import tensorflow as tf

#scipy- statistical analysis tools
from scipy.stats import ttest_1samp
#from scipy import interp

from root.utils.generate_labels import generate_labels

from sklearn.metrics import accuracy_score


def boostrapping_hypothesis_testing(X_train, y_train, X_test, y_test,
                                    n_iterations=100, n_epochs=50):
    '''
    hypothesis testing function
    X_train, y_train, X_test, y_test- the data
    n_iterations- number of bootdtaping iterations
    n_epochs - number of epochs for model's training
    '''

    accuracy = []  ## model accuracy
    roc_msrmnts_fpr = []  ## false positive rate
    roc_msrmnts_tpr = []  ## true positive rate

    # run bootstrap
    for i in range(n_iterations):
        # prepare train and test sets
        X_train, X_test, y_train, y_test = get_train_test(all_subjects_data_reshaped,
                                                          labels, i=i, verbrose=False)
        # fit model
        print('fitting..')
        model.fit(X_train, y_train, validation_split=0.2, epochs=n_epochs)

        # evaluate model
        print('evaluating..')
        y_pred = model.predict(X_test)
        y_test_1d = [i[0] for i in y_test]
        y_pred_1d = [1.0 if i[0] > .5 else 0.0 for i in y_pred]

        fpr, tpr, _ = roc_curve(y_test_1d, y_pred_1d)

        acc_score = accuracy_score(y_test_1d, y_pred_1d)

        accuracy.append(acc_score)
        roc_msrmnts_fpr.append(fpr)
        roc_msrmnts_tpr.append(tpr)

    return accuracy, roc_msrmnts_fpr, roc_msrmnts_tpr




# The data, split between train and test sets.

def get_train_test(X, y, i, verbrose=False):
    '''
    split to train and test and reshape data
    X data
    y labels
    i random state
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.2, random_state=i)

    # Reshapes data to 4D for Hierarchical RNN.
    t_shape = np.array(all_subjects_data_reshaped).shape[1]
    RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

    X_train = np.reshape(X_train, (len(X_train), t_shape, RSN_shape))
    X_test = np.reshape(X_test, (len(X_test), t_shape, RSN_shape))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    if verbrose:
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

    # Converts class vectors to binary class matrices.
    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    return X_train, X_test, y_train, y_test


def test_rnn():
    TIME_SERIES_FOLDER = "/Users/eb/PycharmProjects/neuroNL2/root/bin/output/stopsignal2/time_series/"
    all_subjects_data, labels = generate_labels(TIME_SERIES_FOLDER)

    print('N control:', labels.count(1))
    print('N adhd:', labels.count(5))

    # for i in all_subjects_data:
    #     print(len(i))
    # plt.hist([len(i) for i in all_subjects_data])
    # plt.title('fMRI dataset length variation')
    # plt.xlabel('time stamps')
    # plt.ylabel('#')
    # plt.show()

    max_len_image = np.max([len(i) for i in all_subjects_data])
    print(max_len_image)

    all_subjects_data_reshaped = []
    for subject_data in all_subjects_data:
        # Padding
        N = max_len_image - len(subject_data)
        padded_array = np.pad(subject_data, ((0, N), (0, 0)),
                              'constant', constant_values=(0))
        subject_data = padded_array
        subject_data = np.array(subject_data)
        subject_data.reshape(subject_data.shape[0], subject_data.shape[1], 1)
        all_subjects_data_reshaped.append(subject_data)

    # shape of data

    # 40 subjects
    # 261 time stamps
    # 10 netwroks values
    # (40, 261, 70)

    print(np.array(all_subjects_data_reshaped).shape)
    #(171, 184, 71)

    # create the model

    model = Sequential()

    # LSTM layers -
    # Long Short-Term Memory layer - Hochreiter 1997.
    t_shape = np.array(all_subjects_data_reshaped).shape[1]
    RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

    model.add(LSTM(units=70,  # dimensionality of the output space
                   dropout=0.4,  # Fraction of the units to drop (inputs)
                   recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
                   return_sequences=True,  # return the last state in addition to the output
                   input_shape=(t_shape, RSN_shape)))
    model.add(GRU(units=70,  # dimensionality of the output space
                   dropout=0.4,  # Fraction of the units to drop (inputs)
                   recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
                   return_sequences=True,  # return the last state in addition to the output
                   input_shape=(t_shape, RSN_shape)))

    # model.add(LSTM(units=60,
    #                dropout=0.4,
    #                recurrent_dropout=0.15,
    #                return_sequences=True))
    #
    # model.add(LSTM(units=50,
    #                dropout=0.4,
    #                recurrent_dropout=0.15,
    #                return_sequences=True))
    #
    # model.add(LSTM(units=40,
    #                dropout=0.4,
    #                recurrent_dropout=0.15,
    #                return_sequences=False))

    model.add(GRU(units=60,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(GRU(units=50,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(GRU(units=40,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=False))

    model.add(Dense(units=2,
                    activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['binary_accuracy'])

    print(model.summary())


    ##
    # get_train_test(X, y, i, verbrose=False)
    # X_train, X_test, y_train, y_test = get_train_test(all_subjects_data_reshaped,
    #                                                   labels, i=8, verbrose=True)

    '''
       split to train and test and reshape data
       X data
       y labels
       i random state
       '''

    X = all_subjects_data_reshaped
    y = labels

    #changel labels to 0 1
    print(y)
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0

    for i in range(len(y)):
        if y[i] == 5:
            y[i] = 1

    i = 8
    verbrose = True

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.2, random_state=i)

    # Reshapes data to 4D for Hierarchical RNN.
    t_shape = np.array(all_subjects_data_reshaped).shape[1]
    RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

    X_train = np.reshape(X_train, (len(X_train), t_shape, RSN_shape))
    X_test = np.reshape(X_test, (len(X_test), t_shape, RSN_shape))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    if verbrose:
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

    # Converts class vectors to binary class matrices.
    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=30)

    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    accuracy, roc_msrmnts_fpr, roc_msrmnts_tpr = boostrapping_hypothesis_testing(X_train, y_train, X_test, y_test)

test_rnn()