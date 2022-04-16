from nilearn.connectome import ConnectivityMeasure
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def svm_clustering(pooled_subjects, time_series_filenames, classes):
    # CLUSTERING VIA SVM + CROSS-VALIDATION
    kinds = ['correlation', 'partial correlation', 'tangent']

    n_splits = 15
    test_size = 15
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=0, test_size=test_size)
    pooled_subjects = np.asarray(pooled_subjects)

    print("## DATA ##")
    print("dataset_size: ", len(time_series_filenames))
    print("test_size :", test_size)
    print("n_splits: ", n_splits, "\n")

    #scores = {}
    sensitivity = {}
    specificity = {}
    accuracy = {}
    for kind in kinds:
        print("## ", kind, " ##")
        #scores[kind] = []
        sensitivity[kind]= []
        specificity[kind] = []
        accuracy[kind] = []
        for train, test in cv.split(pooled_subjects, classes):
            # *ConnectivityMeasure* can output the estimated subjects coefficients
            # as a 1D arrays through the parameter *vectorize*.
            connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
            # build vectorized connectomes for subjects in the train set
            connectomes = connectivity.fit_transform(pooled_subjects[train])
            # fit the classifier
            classifier = LinearSVC().fit(connectomes, classes[train])
            # make predictions for the left-out test subjects
            predictions = classifier.predict(
                connectivity.transform(pooled_subjects[test]))

            # print results
            # print("classes: ", classes[test])
            # print("predict: ", predictions)
            # print("accuracy :", accuracy_score(classes[test], predictions), "\n")

            # store the accuracy for this cross-validation fold
            # scores[kind].append(accuracy_score(classes[test], predictions))
            tn, fp, fn, tp = confusion_matrix(classes[test], predictions).ravel()
            sens = tp / (tp + fn)
            sp = tn / (tn + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)

            sensitivity[kind].append(sens)
            specificity[kind].append(sp)
            accuracy[kind].append(acc)

    # mean_scores = [np.mean(scores[kind]) for kind in kinds]
    # scores_std = [np.std(scores[kind]) for kind in kinds]
    mean_sensitivity = [np.mean(sensitivity[kind]) for kind in kinds]
    sensitivity_std = [np.std(sensitivity[kind]) for kind in kinds]
    mean_specificity = [np.mean(specificity[kind]) for kind in kinds]
    specificity_std = [np.std(specificity[kind]) for kind in kinds]
    mean_accuracy = [np.mean(accuracy[kind]) for kind in kinds]
    accuracy_std = [np.std(accuracy[kind]) for kind in kinds]

    print("mean_sensitivity", mean_sensitivity)
    print("sensitivity_std", sensitivity_std)
    print("mean_specificity", mean_specificity)
    print("specificity_std", specificity_std)
    print("mean_sensitivity", mean_sensitivity)
    print("sensitivity_std", sensitivity_std)
    print("mean_accuracy", mean_accuracy)
    print("accuracy_std", accuracy_std)

    #return kinds, scores, mean_scores, scores_std