import os
import numpy
from utils.create_folder import create_folder

def load_time_series(OUTPUT_FOLDER, TIME_SERIES_FOLDER):
 # LOAD TIME-SERIES
    create_folder(OUTPUT_FOLDER)
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

    return pooled_subjects, time_series_filenames

def generate_classes_labels(time_series_filenames):
    #GENERATE LABELS
    classes_list = []
    for time_series_filename in time_series_filenames:
        if "sub-1" in time_series_filename:
            classes_list.append(1)
        if "sub-5" in time_series_filename:
            classes_list.append(5)

    classes = numpy.asarray(classes_list)

    return classes
