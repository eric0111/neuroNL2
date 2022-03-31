from root.clustering import clustering
from root.generate_ica_time_series import generate_ica_time_series
from root.images_cleaner import images_cleaner

def main():
    # CLEAN IMAGES
    FILES  =  "/home/eb/Desktop/test_files/files/"
    CONFOUNDS =  "/home/eb/Desktop/test_files/confounds/"
    OUTPUT_FOLDER = "/home/eb/Desktop/test_files/cleaned/"
    images_cleaner(FILES, CONFOUNDS, OUTPUT_FOLDER)

    # GENERATE TIME-SERIES
    IMAGES_FOLDER = "/home/eb/Desktop/test_files/cleaned/"
    OUTPUT_FOLDER = "./bin/output/test/"
    generate_ica_time_series(IMAGES_FOLDER, OUTPUT_FOLDER)

    # CLUSTERING VIA SVM + CV
    TIME_SERIES_FOLDER = "./bin/output/test/time_series/"
    OUTPUT_FOLDER = "./bin/output/test/"
    clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)

main()