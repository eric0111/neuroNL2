import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from clustering.nn_utils.rnn.networks import add_cnn, add_LSTM, add_GRU


def generate_rnn_model(all_subjects_data_reshaped, CNN, LSTM, GRU):
    model = Sequential()
    t_shape = np.array(all_subjects_data_reshaped).shape[1]
    RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

    if (CNN):
        add_cnn(tf, model, t_shape, RSN_shape)
    if (LSTM):
        add_LSTM(model, t_shape, RSN_shape)
    if (GRU):
        add_GRU(model, t_shape, RSN_shape)

    model.add(Dense(units=2,
                    activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['binary_accuracy'])

    print(model.summary())

    return model
