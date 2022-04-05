import os

import numpy

from root.utils.create_folder import create_folder


def generate_labels(TIME_SERIES_FOLDER):
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
        pooled_subjects.append((numpy.load(time_series_abs_path)).tolist())

    # GENERATE LABELS
    classes_list = []
    for time_series_filename in time_series_filenames:
        if "sub-1" in time_series_filename:
            classes_list.append(1)
        if "sub-5" in time_series_filename:
            classes_list.append(5)

    #classes = np.asarray(classes_list)

    return pooled_subjects, classes_list