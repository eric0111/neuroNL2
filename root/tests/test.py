from root.clustering import clustering
from root.generate_ica_time_series import generate_ica_time_series
from root.images_cleaner import images_cleaner

IMAGES_FOLDER = "/Users/eb/Desktop/stopsignal_cleaned/"
OUTPUT_FOLDER = "../bin/output/stopsignal/"
generate_ica_time_series(IMAGES_FOLDER, OUTPUT_FOLDER)

TIME_SERIES_FOLDER = "../bin/output/stopsignal/time_series/"
OUTPUT_FOLDER = "../bin/output/stopsignal/"
clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)

# FILES  =  "/Users/eb/Desktop/bart/"
# CONFOUNDS =  "/Users/eb/Desktop/bart_confounds/"
# OUTPUT_FOLDER = "/Users/eb/Desktop/bart_cleaned/"
# images_cleaner(FILES, CONFOUNDS, OUTPUT_FOLDER)
#
# IMAGES_FOLDER = "/Users/eb/Desktop/bart_cleaned/"
# OUTPUT_FOLDER = "../bin/output/bart/"
# generate_ica_time_series(IMAGES_FOLDER, OUTPUT_FOLDER)
#
# TIME_SERIES_FOLDER = "../bin/output/bart/time_series/"
# OUTPUT_FOLDER = "../bin/output/bart/"
# clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)
