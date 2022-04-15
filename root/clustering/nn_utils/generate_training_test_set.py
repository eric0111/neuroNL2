from sklearn.model_selection import train_test_split


def generate_training_test_set(all_subjects_data_reshaped, labels):
    X = all_subjects_data_reshaped
    y = labels

    # change labels to 0 1
    print(y)
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0

    for i in range(len(y)):
        if y[i] == 5:
            y[i] = 1

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.1, random_state=8)

    return X_train, X_test, y_train, y_test