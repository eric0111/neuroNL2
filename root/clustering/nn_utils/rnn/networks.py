from keras.layers import Convolution1D, LSTM, GRU

def add_cnn(tf, model, t_shape, RSN_shape):
    # 3d convolution
    # https://github.com/bsplku/3dcnn4fmri/blob/master/Python_code/3dcnn_fmri_demo.ipynb
    model.add(Convolution1D(input_shape=(t_shape, RSN_shape),
                            filters=32,
                            kernel_size=(3),
                            activation=tf.nn.relu))

def add_LSTM(model, t_shape, RSN_shape):
    model.add(LSTM(units=70,  # dimensionality of the output space
                   dropout=0.4,  # Fraction of the units to drop (inputs)
                   recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
                   return_sequences=True,  # return the last state in addition to the output
                   input_shape=(t_shape, RSN_shape)))

    model.add(LSTM(units=60,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(LSTM(units=50,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(LSTM(units=40,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=False))

def add_GRU(model, t_shape, RSN_shape):
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


