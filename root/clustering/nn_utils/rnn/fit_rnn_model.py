import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def fit_rnn_model(all_subjects_data_reshaped, model, X_train, X_test, y_train, y_test):
    verbose = False

    # Reshapes data to 4D for Hierarchical RNN.
    t_shape = np.array(all_subjects_data_reshaped).shape[1]
    RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

    X_train = np.reshape(X_train, (len(X_train), t_shape, RSN_shape))
    X_test = np.reshape(X_test, (len(X_test), t_shape, RSN_shape))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    if verbose:
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

    # Converts class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_test = tf.keras.utils.to_categorical(y_test, 2)

    history = model.fit(X_train, y_train, validation_split=0.1, epochs=100,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-5)
                        ])

    return history, X_test, y_test
