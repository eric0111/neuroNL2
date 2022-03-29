from root.clustering import clustering
from root.generate_ica_time_series import generate_ica_time_series
from root.images_cleaner import images_cleaner


def main():
    #CLEAN IMAGES
    FILES  =  "/Users/eb/Desktop/test_files/files/"
    CONFOUNDS =  "/Users/eb/Desktop/test_files/confounds/"
    CLEANED_FOLDER = "/Users/eb/Desktop/test_files/cleaned/"
    images_cleaner(FILES, CONFOUNDS, CLEANED_FOLDER)

    #GENERATE TIME-SERIES
    IMAGES_FOLDER = "/Users/eb/Desktop/stopsignal_cleaned/"
    OUTPUT_FOLDER = "./bin/output/stopsignal/"
    generate_ica_time_series(IMAGES_FOLDER, OUTPUT_FOLDER)

    #CLUSTERING VIA SVM + CV
    TIME_SERIES_FOLDER = "./bin/output/test/time_series/"
    OUTPUT_FOLDER = "./bin/output/test/"
    clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)

main()