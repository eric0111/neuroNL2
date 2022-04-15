from clustering.nn_utils.data_augmentation import data_augmentation
from clustering.nn_utils.generate_data_reshaped_and_labels import generate_data_reshaped_and_labels
from clustering.nn_utils.generate_results import generate_results
from clustering.nn_utils.generate_training_test_set import generate_training_test_set
from clustering.nn_utils.rnn.fit_rnn_model import fit_rnn_model
from clustering.nn_utils.rnn.generate_rnn_model import generate_rnn_model


def rnn(TIME_SERIES_FOLDER,  CNN, LSTM, GRU):
    all_subjects_data_reshaped, labels = generate_data_reshaped_and_labels(TIME_SERIES_FOLDER)
    X_train, X_test, y_train, y_test = generate_training_test_set(all_subjects_data_reshaped, labels)
    X_train, X_test, y_train, y_test = data_augmentation(X_train, X_test, y_train, y_test)

    model = generate_rnn_model(all_subjects_data_reshaped, CNN, LSTM, GRU)
    history, X_test, y_test = fit_rnn_model(all_subjects_data_reshaped, model, X_train, X_test, y_train, y_test)
    generate_results(history, model, X_test, y_test)