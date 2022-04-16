from clustering.nn_utils.data_augmentation import data_augmentation
from clustering.nn_utils.generate_data_reshaped_and_labels import generate_data_reshaped_and_labels
from clustering.nn_utils.generate_results import generate_results
from clustering.nn_utils.generate_training_test_set import generate_training_test_set
from clustering.nn_utils.transformers.fit_transformers_model import fit_transformer_model
from clustering.nn_utils.transformers.generate_transformers_model import generate_tranformer_model


def transformers(TIME_SERIES_FOLDER):
    all_subjects_data_reshaped, labels = generate_data_reshaped_and_labels(TIME_SERIES_FOLDER)
    X_train, X_test, y_train, y_test = generate_training_test_set(all_subjects_data_reshaped, labels)
    X_train, X_test, y_train, y_test = data_augmentation(X_train, X_test, y_train, y_test)


    model, X_test, y_test, X_train, y_train = generate_tranformer_model(all_subjects_data_reshaped, X_train, X_test, y_train, y_test)
    history = fit_transformer_model(model, X_train, y_train)
    generate_results(history, model, X_test, y_test)