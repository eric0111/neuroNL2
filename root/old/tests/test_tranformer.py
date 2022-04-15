import numpy
import numpy as np

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

def mini_test():
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    print(x_train[0])
    print("\n")
    print(y_train)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    n_classes = len(np.unique(y_train))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    import numpy as np
    import matplotlib.pyplot as plt
    # matplotlib inline
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

# data augmentation deep learning
# tsaug per data augmentation
# giocare con la network

# kaggle per esecuzione
#
import keras
from keras import layers


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    n_classes = 2  # todo added
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)



def test_transformer():
    import numpy as np
    TIME_SERIES_FOLDER = "/home/eb/PycharmProjects/neuroNL2/root/bin/output/stopsignal2/time_series_dict/"
    # TIME_SERIES_FOLDER = "/home/eb/PycharmProjects/neuroNL2/root/bin/output/bart2/time_series_dict/"
    all_subjects_data, labels = generate_labels(TIME_SERIES_FOLDER)

    print('N control:', labels.count(1))
    print('N adhd:', labels.count(5))

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
    # (171, 184, 71)

    # create the model
    X = all_subjects_data_reshaped
    y = labels

    # changel labels to 0 1
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

    n_classes = 2

    from tensorflow import keras
    from keras import layers

    input_shape = X_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    # model.compile(
    #     loss="sparse_categorical_crossentropy",
    #     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #     metrics=["sparse_categorical_accuracy"],
    # )
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['binary_accuracy'])

    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-5)]

    print(y_train)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    #
    # history = model.fit(X_train, y_train, validation_split=0.1, epochs=100,
    #                     callbacks=[
    #                         EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
    #                         ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-5)
    #                     ])

   # model.evaluate(X_test, numpy.asarray(y_test), verbose=1)
    from matplotlib import pyplot as plt
    # # summarize history for accuracy
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

test_transformer()