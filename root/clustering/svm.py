from clustering.svm_utils.load_time_series_and_labels import load_time_series, generate_classes_labels
from clustering.svm_utils.svm_clustering import svm_clustering
from clustering.svm_utils.svm_results import svm_results


def svm(TIME_SERIES_FOLDER, OUTPUT_FOLDER):
    pooled_subjects, time_series_filenames = load_time_series(OUTPUT_FOLDER, TIME_SERIES_FOLDER)
    classes = generate_classes_labels(time_series_filenames)

    svm_clustering(pooled_subjects, time_series_filenames, classes)
    #svm_results(OUTPUT_FOLDER, kinds, mean_scores, scores_std)

