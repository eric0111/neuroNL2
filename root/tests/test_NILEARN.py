from root.clustering import clustering
from root.generate_ica_time_series_NILEARN import generate_ica_time_series_NILEARN
import time

t = time.time()
IMAGES_FOLDER = "/home/eb/Desktop/stopsignal/"
CONFOUNDS = "/home/eb/Desktop/stopsignal_confounds/"
OUTPUT_FOLDER = "../bin/output/stopsignal2/"
generate_ica_time_series_NILEARN(IMAGES_FOLDER, OUTPUT_FOLDER, CONFOUNDS)
elapsed = time.time() - t
print("generate_ica_time_series2 - stopsignal: ", elapsed)
#generate_ica_time_series: 1558.769073009491 -- 26 mins

t = time.time()
TIME_SERIES_FOLDER = "../bin/output/stopsignal2/time_series/"
OUTPUT_FOLDER = "../bin/output/stopsignal2/"
clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)
elapsed = time.time() - t
print("clustering2 - stopsignal: ", elapsed)
#clustering: 58.56087112426758 -- 59 secs

t = time.time()
IMAGES_FOLDER = "/home/eb/Desktop/bart/"
OUTPUT_FOLDER = "../bin/output/bart2/"
CONFOUNDS = "/home/eb/Desktop/bart_confounds/"
generate_ica_time_series_NILEARN(IMAGES_FOLDER, OUTPUT_FOLDER, CONFOUNDS)
elapsed = time.time() - t
print("generate_ica_time_series2 - bart: ", elapsed)
#generate_ica_time_series2 - bart:  2494.587844133377 -- 42 mins

t = time.time()
TIME_SERIES_FOLDER = "../bin/output/bart2/time_series/"
OUTPUT_FOLDER = "../bin/output/bart2/"
clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)
elapsed = time.time() - t
print("clustering2 - bart: ", elapsed)
#clustering2 - bart:  33.32945895195007 -- 33 secs