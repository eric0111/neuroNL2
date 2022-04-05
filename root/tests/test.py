from root.clustering import clustering
from root.generate_ica_time_series import generate_ica_time_series
from root.generate_ica_time_series_NILEARN import generate_ica_time_series_NILEARN
from root.images_cleaner import images_cleaner
import time
#TIME - stopsignal
#generate_ica_time_series:  2562.8496882915497 - 43 mins
#clustering:  36.4996874332428 - 35 secs

#TIME - bart
#images_cleaner:  6574.576639652252 - 110 mins
#generate_ica_time_series:  2420.6614661216736 - 41 mins
#clustering:  30.42004108428955 - 31 secs

# ICA IMGS FOR COMPONENTS
# images = nilearn.image.load_img(images_abs_paths[0:24] + images_abs_paths[146:170])
    ## ok stopsignal, ko bart
    #images = nilearn.image.load_img(images_abs_paths[0:22] + images_abs_paths[-23:-1])
    ## ko bart
    #images = nilearn.image.load_img(images_abs_paths[0:15] + images_abs_paths[-16:-1])
    ## ko bart

#STOPSIGNAL
t = time.time()
FILES  =  "/home/eb/Desktop/stopsignal/"
CONFOUNDS =  "/home/eb/Desktop/stopsignal_confounds/"
OUTPUT_FOLDER = "/home/eb/Desktop/stopsignal_cleaned/"
images_cleaner(FILES, CONFOUNDS, OUTPUT_FOLDER)
elapsed = time.time() - t
print("images_cleaner: ", elapsed)
#images_cleaner:  6574.576639652252 - 110 mins

t = time.time()
IMAGES_FOLDER = "/home/eb/Desktop/stopsignal_cleaned/"
OUTPUT_FOLDER = "../bin/output/stopsignal/"
generate_ica_time_series(IMAGES_FOLDER, OUTPUT_FOLDER)
elapsed = time.time() - t
print("generate_ica_time_series: ", elapsed)
#generate_ica_time_series:  2562.8496882915497 - 43 mins

t = time.time()
TIME_SERIES_FOLDER = "../bin/output/stopsignal2/time_series/"
OUTPUT_FOLDER = "../bin/output/stopsignal2/"
clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)
elapsed = time.time() - t
print("clustering: ", elapsed)
#clustering:  36.4996874332428 - 35 secs


#BART
t = time.time()
FILES  =  "/home/eb/Desktop/bart/"
CONFOUNDS =  "/home/eb/Desktop/bart_confounds/"
OUTPUT_FOLDER = "/home/eb/Desktop/bart_cleaned/"
images_cleaner(FILES, CONFOUNDS, OUTPUT_FOLDER)
elapsed = time.time() - t
print("images_cleaner: ", elapsed)
# #images_cleaner:  6574.576639652252 - 110 mins

t = time.time()
IMAGES_FOLDER = "/home/eb/Desktop/bart_cleaned/"
OUTPUT_FOLDER = "../bin/output/bart/"
generate_ica_time_series(IMAGES_FOLDER, OUTPUT_FOLDER)
elapsed = time.time() - t
print("generate_ica_time_series: ", elapsed)

t = time.time()
TIME_SERIES_FOLDER = "../bin/output/bart/time_series/"
OUTPUT_FOLDER = "../bin/output/bart/"
clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)
elapsed = time.time() - t
print("clustering: ", elapsed)