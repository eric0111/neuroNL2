import numpy
from matplotlib import pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import numpy as np
import os

def clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER):
    # LOAD TIME-SERIES
    time_series_filenames = os.listdir(TIME_SERIES_FOLDER)
    try:
        time_series_filenames.remove(".DS_Store")
    except Exception as e:
        print(e)
    time_series_filenames.sort()

    time_series_abs_paths = []
    for time_series in time_series_filenames:
        time_series_abs_paths.append(TIME_SERIES_FOLDER + time_series)

    pooled_subjects = []
    for time_series_abs_path in time_series_abs_paths:
        pooled_subjects.append(numpy.load(time_series_abs_path))

    #GENERATE LABELS
    classes_list = []
    for time_series_filename in time_series_filenames:
        if "sub-1" in time_series_filename:
            classes_list.append(1)
        if "sub-5" in time_series_filename:
            classes_list.append(5)

    classes = np.asarray(classes_list)
    print(classes)

    #CLUSTERING VIA SVM + CROSS-VALIDATION
    #kinds = ['correlation', 'partial correlation', 'tangent']
    kinds = ['correlation']

    cv = StratifiedShuffleSplit(n_splits=6, random_state=0, test_size=2)
    pooled_subjects = np.asarray(pooled_subjects)

    scores = {}
    for kind in kinds:
        scores[kind] = []
        for train, test in cv.split(pooled_subjects, classes):
            print(train, test)
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
            # print(predictions)
            # print(classes[test])
            # store the accuracy for this cross-validation fold
            scores[kind].append(accuracy_score(classes[test], predictions))

    mean_scores = [np.mean(scores[kind]) for kind in kinds]
    scores_std = [np.std(scores[kind]) for kind in kinds]

    plt.figure(figsize=(6, 4))
    positions = np.arange(len(kinds)) * .1 + .1
    plt.barh(positions, mean_scores, align='center', height=.05, xerr=scores_std)
    yticks = [k.replace(' ', '\n') for k in kinds]
    plt.yticks(positions, yticks)
    plt.gca().grid(True)
    plt.gca().set_axisbelow(True)
    plt.gca().axvline(.8, color='red', linestyle='--')
    plt.xlabel('Classification accuracy\n(red line = chance level)')
    plt.tight_layout()
    plt.savefig("test.png")
