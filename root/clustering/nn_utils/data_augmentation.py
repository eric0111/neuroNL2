from imblearn.over_sampling import SMOTE
import numpy as np

def data_augmentation(X_train, X_test, y_train, y_test):
    ## data augmentation - training set
    sm = SMOTE(random_state=42)

    nsamples, nx, ny = np.asarray(X_train).shape
    X_train = np.asarray(X_train).reshape((nsamples, nx * ny))
    X_train, y_train = sm.fit_resample(X_train, y_train)


    ## data augmentation - test set
    sm = SMOTE(random_state=42)

    nsamples, nx, ny = np.asarray(X_test).shape
    X_test = np.asarray(X_test).reshape((nsamples, nx * ny))
    X_test, y_test = sm.fit_resample(X_test, y_test)

    return X_train, X_test, y_train, y_test