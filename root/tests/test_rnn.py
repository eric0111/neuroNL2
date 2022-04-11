import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

import tensorflow as tf
from root.utils.generate_labels import generate_labels

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

#data augmentation deep learning
#tsaug per data augmentation
#giocare con la network

#kaggle per esecuzione
#



def test_rnn():
    TIME_SERIES_FOLDER = "/home/eb/PycharmProjects/neuroNL2/root/bin/output/stopsignal2/time_series_dict/"
    #TIME_SERIES_FOLDER = "/home/eb/PycharmProjects/neuroNL2/root/bin/output/bart2/time_series_dict/"
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

    #3d convolution
    #https://github.com/bsplku/3dcnn4fmri/blob/master/Python_code/3dcnn_fmri_demo.ipynb
    # model.add(Convolution1D(input_shape=(t_shape, RSN_shape),
    #                         filters=32,
    #                         kernel_size=(3),
    #                         activation=tf.nn.relu))
    # ###

    # model.add(LSTM(units=70,  # dimensionality of the output space
    #                dropout=0.4,  # Fraction of the units to drop (inputs)
    #                recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
    #                return_sequences=True,  # return the last state in addition to the output
    #                input_shape=(t_shape, RSN_shape)))

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

    model.add(GRU(units=70,  # dimensionality of the output space
                  dropout=0.4,  # Fraction of the units to drop (inputs)
                  recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
                  return_sequences=True,  # return the last state in addition to the output
                  input_shape=(t_shape, RSN_shape)))

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

    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate= 1e-3,
    #     decay_steps=10000,
    #     decay_rate=0.99
    # )
    # model.compile(loss='binary_crossentropy',
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    #               metrics=['binary_accuracy'])

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
                                                        y, test_size=0.1, random_state=i)

    ## data augmentation - training set
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)

    nsamples, nx, ny = np.asarray(X_train).shape
    X_train = np.asarray(X_train).reshape((nsamples, nx * ny))

    X_train, y_train = sm.fit_resample(X_train, y_train)
    ##

    ## data augmentation - test set
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)

    nsamples, nx, ny = np.asarray(X_test).shape
    X_test = np.asarray(X_test).reshape((nsamples, nx * ny))

    X_test, y_test = sm.fit_resample(X_test, y_test)


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
    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_test = tf.keras.utils.to_categorical(y_test, 2)

    history = model.fit(X_train, y_train, validation_split=0.1, epochs=100,
                        callbacks= [
                            EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-5)
                        ])

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

    #model.fit(X_train, y_train, epochs=30)

    # evaluate model
    print('evaluating..')
    y_pred = model.predict(X_test)
    y_test_1d = [i[0] for i in y_test]
    y_pred_1d = [1.0 if i[0] > .5 else 0.0 for i in y_pred]
    acc_score = accuracy_score(y_test_1d, y_pred_1d)

    print(acc_score)

test_rnn()